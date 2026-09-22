#include "diffusion.h"

#include "log.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

bool diffusion_model_supports_entropy_bounded(const llama_model * model) {
    if (!model) {
        return false;
    }

    char architecture[32] = {};
    char causal_attention[8] = {};
    char shift_logits[8] = {};
    if (llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) < 0 ||
            llama_model_meta_val_str(model, "deepseek2.attention.causal", causal_attention, sizeof(causal_attention)) < 0 ||
            llama_model_meta_val_str(model, "diffusion.shift_logits", shift_logits, sizeof(shift_logits)) < 0) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    return strcmp(architecture, "deepseek2") == 0 &&
           strcmp(causal_attention, "false") == 0 &&
           strcmp(shift_logits, "false") == 0 &&
           llama_vocab_mask(vocab) != LLAMA_TOKEN_NULL;
}

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case DIFFUSION_ALGORITHM_CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;  // Selected token probability

        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;  // Higher entropy = lower confidence
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;

        case DIFFUSION_ALGORITHM_RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);  // Random confidence
            }

        case DIFFUSION_ALGORITHM_ORIGIN:
            return cur_p.data[cur_p.selected].p;

        default:
            return 0.0f;
    }
}

// Unified transfer count calculation function
static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        diffusion_transfer_schedule  schedule,
                                        float                        eps,
                                        const std::vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED:
            {
                float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }

        case DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / (total_steps - step);  // Fallback

        default:
            return remaining_masked / (total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

static int decode_block(
        llama_context *       ctx,
        llama_batch &         batch,
        const llama_token *   tokens,
        int32_t               block_start,
        int32_t               block_length,
        llama_token           mask_token_id,
        bool                  transient) {
    batch.n_tokens = block_length;
    for (int32_t i = 0; i < block_length; ++i) {
        batch.token[i]     = tokens[block_start + i];
        batch.pos[i]       = block_start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = transient ? tokens[block_start + i] == mask_token_id : i == block_length - 1;
    }
    const int ret = llama_decode(ctx, batch);
    if (ret == 0 && !transient) {
        // The next decode consumes this block without reading its logits.
        llama_synchronize(ctx);
    }
    return ret;
}

static float entropy_and_argmax(
        const float * logits,
        int32_t       n_vocab,
        llama_token   mask_token_id,
        llama_token & argmax_token) {
    float max_logit = -INFINITY;
    argmax_token = LLAMA_TOKEN_NULL;

    for (int32_t token_id = 0; token_id < n_vocab; ++token_id) {
        if (token_id == mask_token_id) {
            continue;
        }
        if (logits[token_id] > max_logit) {
            max_logit = logits[token_id];
            argmax_token = token_id;
        }
    }

    float sum_exp = 0.0f;
    float sum_exp_logit = 0.0f;
    for (int32_t token_id = 0; token_id < n_vocab; ++token_id) {
        if (token_id == mask_token_id) {
            continue;
        }
        const float weight = expf(logits[token_id] - max_logit);
        sum_exp += weight;
        sum_exp_logit += weight * logits[token_id];
    }

    return logf(sum_exp) + max_logit - sum_exp_logit / sum_exp;
}

static llama_token sample_candidate(
        const float *                  logits,
        int32_t                        n_vocab,
        llama_token                    mask_token_id,
        llama_sampler *                sampler,
        std::vector<llama_token_data> & candidates) {
    for (int32_t token_id = 0; token_id < n_vocab; ++token_id) {
        candidates[token_id].id    = token_id;
        candidates[token_id].logit = token_id == mask_token_id ? -INFINITY : logits[token_id];
        candidates[token_id].p     = 0.0f;
    }

    llama_token_data_array cur_p = {
        candidates.data(),
        candidates.size(),
        -1,
        false,
    };
    llama_sampler_apply(sampler, &cur_p);
    return cur_p.data[cur_p.selected].id;
}

static bool entropy_bounded_step(
        llama_context *                  ctx,
        llama_token *                    output_tokens,
        int32_t                          block_start,
        int32_t                          block_length,
        const diffusion_params &         params,
        int32_t                          n_vocab,
        llama_sampler *                  sampler,
        std::vector<llama_token_data> &  candidates) {
    struct token_entropy {
        float entropy;
        int32_t block_pos;
        llama_token token;
    };

    std::vector<token_entropy> masked;
    masked.reserve(block_length);

    const float * all_logits = llama_get_logits(ctx);
    if (!all_logits) {
        LOG_ERR("%s: logits are not available\n", __func__);
        return false;
    }

    int32_t logits_row = 0;
    for (int32_t i = 0; i < block_length; ++i) {
        if (output_tokens[block_start + i] != params.mask_token_id) {
            continue;
        }

        // llama_get_logits() packs rows with batch.logits=true in batch order.
        const float * logits = all_logits + (size_t) logits_row++ * n_vocab;
        llama_token candidate = LLAMA_TOKEN_NULL;
        const float entropy = entropy_and_argmax(logits, n_vocab, params.mask_token_id, candidate);

        if (params.temperature > 0.0f) {
            candidate = sample_candidate(logits, n_vocab, params.mask_token_id, sampler, candidates);
        }
        masked.push_back({ entropy, i, candidate });
    }

    if (masked.empty()) {
        return true;
    }

    std::sort(masked.begin(), masked.end(), [](const token_entropy & a, const token_entropy & b) {
        if (a.entropy != b.entropy) {
            return a.entropy < b.entropy;
        }
        return a.block_pos < b.block_pos;
    });

    int32_t reveal_count = 1;
    double entropy_budget = 0.0;
    for (size_t i = 0; i + 1 < masked.size(); ++i) {
        entropy_budget += masked[i].entropy;
        if (entropy_budget <= params.gamma) {
            reveal_count++;
        } else {
            break;
        }
    }

    for (int32_t i = 0; i < reveal_count; ++i) {
        const token_entropy & item = masked[i];
        output_tokens[block_start + item.block_pos] = item.token;
    }

    return true;
}

static int32_t find_token_position(
        const llama_token * tokens,
        int32_t             start,
        int32_t             end,
        llama_token         token) {
    const llama_token * it = std::find(tokens + start, tokens + end, token);
    return it == tokens + end ? -1 : it - tokens;
}

static void diffusion_generate_entropy_bounded(
        llama_context *          ctx,
        const llama_token *      input_tokens,
        llama_token *            output_tokens,
        int32_t                  n_input,
        const diffusion_params & params,
        int32_t &                n_generated) {
    if (params.block_length <= 0 || params.steps <= 0 || params.max_length <= 0 ||
            !std::isfinite(params.gamma) || params.gamma < 0.0f || n_input < 0 || n_input > params.max_length ||
            params.max_new_tokens < -1 || params.max_new_tokens > params.max_length - n_input) {
        LOG_ERR("%s: invalid block diffusion dimensions\n", __func__);
        n_generated = 0;
        return;
    }

    const int32_t max_block_aligned_length = (params.max_length / params.block_length) * params.block_length;
    const int32_t max_new_tokens = params.max_new_tokens < 0 ? max_block_aligned_length - n_input : params.max_new_tokens;
    if (max_new_tokens == 0) {
        std::copy(input_tokens, input_tokens + n_input, output_tokens);
        n_generated = 0;
        return;
    }
    const int32_t total_length = n_input + max_new_tokens;
    const int32_t decode_length = ((total_length + params.block_length - 1) / params.block_length) * params.block_length;
    if (max_new_tokens < 0 || decode_length > params.max_length) {
        LOG_ERR("%s: requested sequence does not fit the context\n", __func__);
        n_generated = 0;
        return;
    }
    const llama_model * model = llama_get_model(ctx);
    if (!diffusion_model_supports_entropy_bounded(model) || params.shift_logits) {
        LOG_ERR("%s: model does not support cached entropy-bounded block generation\n", __func__);
        n_generated = 0;
        return;
    }
    if (llama_n_ubatch(ctx) < (uint32_t) params.block_length) {
        LOG_ERR("%s: non-causal block generation requires n_ubatch >= block_length (%u < %d)\n",
                __func__, llama_n_ubatch(ctx), params.block_length);
        n_generated = 0;
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token model_mask_token_id = llama_vocab_mask(vocab);
    if (params.mask_token_id != model_mask_token_id) {
        LOG_ERR("%s: mask token id does not match the model (%d != %d)\n",
                __func__, params.mask_token_id, model_mask_token_id);
        n_generated = 0;
        return;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token eos_token_id = llama_vocab_eos(vocab);

    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + decode_length, params.mask_token_id);

    llama_set_causal_attn(ctx, false);
    llama_memory_t memory = llama_get_memory(ctx);
    llama_synchronize(ctx);
    if (!llama_memory_seq_rm(memory, 0, -1, -1)) {
        LOG_ERR("%s: failed to reset sequence memory\n", __func__);
        n_generated = 0;
        return;
    }
    llama_batch batch = llama_batch_init(params.block_length, 0, 1);

    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));
    std::vector<llama_token_data> candidates(params.temperature > 0.0f ? n_vocab : 0);

    const int32_t num_blocks = decode_length / params.block_length;
    const int32_t num_prefill_blocks = n_input / params.block_length;
    int32_t global_step = 0;
    int32_t generated_end = n_input + max_new_tokens;

    for (int32_t block_idx = 0; block_idx < num_prefill_blocks; ++block_idx) {
        const int32_t block_start = block_idx * params.block_length;
        if (decode_block(ctx, batch, output_tokens, block_start, params.block_length, params.mask_token_id, false) != 0) {
            LOG_ERR("%s: failed to prefill block %d\n", __func__, block_idx);
            goto fail;
        }
    }

    for (int32_t block_idx = num_prefill_blocks; block_idx < num_blocks; ++block_idx) {
        const int32_t block_start = block_idx * params.block_length;
        const int32_t block_end = block_start + params.block_length;

        for (int32_t step = 0; step < params.steps; ++step) {
            if (find_token_position(output_tokens, block_start, block_end, params.mask_token_id) < 0) {
                break;
            }

            if (params.step_callback) {
                const int32_t total_steps = (num_blocks - num_prefill_blocks) * params.steps;
                if (!params.step_callback(global_step, total_steps, output_tokens, decode_length, params.step_callback_user_data)) {
                    goto fail;
                }
            }
            global_step++;

            if (decode_block(ctx, batch, output_tokens, block_start, params.block_length, params.mask_token_id, true) != 0) {
                LOG_ERR("%s: failed to decode block %d at step %d\n", __func__, block_idx, step);
                goto fail;
            }
            if (!entropy_bounded_step(ctx, output_tokens, block_start, params.block_length, params, n_vocab, sampler, candidates)) {
                goto fail;
            }

            if (!llama_memory_seq_rm(memory, 0, block_start, block_end)) {
                LOG_ERR("%s: failed to roll back block %d\n", __func__, block_idx);
                goto fail;
            }

            if (eos_token_id != LLAMA_TOKEN_NULL) {
                const int32_t eos_pos = find_token_position(output_tokens, n_input, block_end, eos_token_id);
                if (eos_pos >= 0 && find_token_position(output_tokens, n_input, eos_pos + 1, params.mask_token_id) < 0) {
                    generated_end = std::min(generated_end, eos_pos + 1);
                    goto cleanup;
                }
            }
        }

        if (decode_block(ctx, batch, output_tokens, block_start, params.block_length, params.mask_token_id, false) != 0) {
            LOG_ERR("%s: failed to commit block %d\n", __func__, block_idx);
            goto fail;
        }

        if (eos_token_id != LLAMA_TOKEN_NULL) {
            const int32_t eos_pos = find_token_position(output_tokens, n_input, block_end, eos_token_id);
            if (eos_pos >= 0) {
                generated_end = std::min(generated_end, eos_pos + 1);
                break;
            }
        }
    }

    {
        const int32_t mask_pos = find_token_position(output_tokens, n_input, generated_end, params.mask_token_id);
        if (mask_pos >= 0) {
            generated_end = mask_pos;
        }
    }
    goto cleanup;

fail:
    llama_synchronize(ctx);
    if (!llama_memory_seq_rm(memory, 0, -1, -1)) {
        LOG_ERR("%s: failed to reset sequence memory after generation failure\n", __func__);
    }
    generated_end = n_input;

cleanup:
    n_generated = std::max(0, generated_end - n_input);
    llama_batch_free(batch);
    llama_sampler_free(sampler);
}

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    if (params.algorithm == DIFFUSION_ALGORITHM_ENTROPY_BOUNDED) {
        diffusion_generate_entropy_bounded(ctx, input_tokens, output_tokens, n_input, params, n_generated);
        return;
    }

    const llama_model * model = llama_get_model(ctx);

    // Initialize with input and pad with mask tokens
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);

    // Setup sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    // Pre-allocate buffers for CFG if needed
    int32_t                  logits_size = n_vocab * params.max_length;
    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        cond_logits_buffer.resize(logits_size);
        un_x_buffer.resize(params.max_length);
    }

    // For block-based processing
    std::vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks      = 1;
    int32_t              steps_per_block = params.steps;

    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        GGML_ASSERT(params.max_length % params.block_length == 0);
        num_blocks = params.max_length / params.block_length;
        GGML_ASSERT(params.steps % num_blocks == 0);
        steps_per_block = params.steps / num_blocks;
    }

    std::vector<float> confidence(params.max_length);

    int64_t total_sampling_time = 0;
    int64_t total_time          = 0;
    int64_t time_start          = ggml_time_us();

    for (int block_num = 0; block_num < num_blocks; block_num++) {
        int32_t block_start = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ? n_input + block_num * params.block_length : 0;
        int32_t block_end   = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ?
                                  std::min(n_input + (block_num + 1) * params.block_length, params.max_length) :
                                  params.max_length;

        // Count masked tokens in current block for block-based processing
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for (int32_t step = 0; step < steps_per_block; step++) {
            int32_t global_step = block_num * steps_per_block + step;

            if (params.step_callback) {
                if (!params.step_callback(
                        global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
                    break;
                }
            }

            // Setup batch
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = output_tokens[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }

            float * logits = nullptr;

            if (params.cfg_scale > 0.0f) {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                    break;
                }
                float * cond_logits_ptr = llama_get_logits(ctx);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                // Unconditional generation (mask input)
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i] = un_x_buffer[i];
                }
                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate unconditional");
                    break;
                }
                float * uncond_logits = llama_get_logits(ctx);

                // Apply CFG
                for (int32_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }
                logits = cond_logits_buffer.data();
            } else {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                    break;
                }
                logits = llama_get_logits(ctx);
            }

            if (!logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                break;
            }

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                if (params.shift_logits) {
                    return pos == 0 ? logits : logits + (pos - 1) * n_vocab;
                }
                return logits + pos * n_vocab;
            };

            int64_t time_start_sampling = ggml_time_us();

            mask_positions.clear();
            for (int32_t i = 0; i < params.max_length; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    // For block-based, only consider current block
                    if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                break;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        const float * pos_logits = get_logits_for_pos(pos);
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].id    = token_id;
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            (size_t) n_vocab,
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected].id;
                    }
                }
            } else {
                std::vector<std::pair<float, int32_t>> confidences;
                std::vector<llama_token>               sampled_tokens(mask_positions.size());

                for (size_t i = 0; i < mask_positions.size(); i++) {
                    int32_t       pos        = mask_positions[i];
                    const float * pos_logits = get_logits_for_pos(pos);

                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                        candidates[token_id].id    = token_id;
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        candidates.size(),
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    llama_token sampled_token = cur_p.data[cur_p.selected].id;

                    float conf = calculate_confidence(cur_p, params.algorithm, rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.emplace_back(conf, i);
                }

                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                if (transfer_count > 0) {
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + std::min(transfer_count, (int32_t) confidences.size()),
                                          confidences.end(),
                                          [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                              if (a.first != b.first) {
                                                  return a.first > b.first;
                                              }
                                              return a.second < b.second;
                                          });

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            int32_t mask_idx   = confidences[i].second;
                            int32_t pos        = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            float conf_logit = confidences[i].first / params.alg_temp;
                            conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                        }

                        llama_token_data_array conf_array = {
                            conf_candidates.data(),
                            conf_candidates.size(),
                            -1,
                            false,
                        };

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            llama_sampler_apply(dist_sampler, &conf_array);
                            int32_t selected_idx = conf_array.selected;
                            int32_t mask_idx     = selected_idx;
                            int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]   = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                }
            }

            int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
        }
    }

    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    LOG_INF("\ntotal time: %0.2fms, time per step: %0.2fms, sampling time per step: %0.2fms\n",
            total_time / 1000.0,
            total_time / 1000.0 / params.steps,
            total_sampling_time / 1000.0 / params.steps);

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    n_generated = params.max_length;
}
