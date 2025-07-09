import torch
from verl.utils.fsdp_utils import (
    offload_fsdp_param_and_grad,
    offload_fsdp_optimizer,
    load_fsdp_param_and_grad,
    load_fsdp_optimizer
)
from verl import DataProto
from verl.utils.debug import log_gpu_memory_usage
from codetiming import Timer

import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def lookahead_update(
    actor_module,
    actor_optimizer,
    update_fn,
    data,
    num_inner_steps,
    step_size,
    device,
    offload_param,
    offload_grad,
    offload_optimizer,
    sharding_manager,
    flops_counter,
    lr_scheduler,
    config,
    world_size
):
    """
    Perform lookahead-style update on the PPO actor.

    Args:
        actor_module: the wrapped FSDP actor module
        actor_optimizer: the optimizer
        update_fn: function to call for normal PPO update (e.g., actor.update_policy)
        data: training data (DataProto)
        num_inner_steps: how many inner PPO updates to run on the same data
        step_size: final lookahead step size
        offload_param: whether FSDP param is offloaded
        offload_grad: whether FSDP grad is offloaded
        offload_optimizer: whether optimizer is offloaded
        sharding_manager: Ulysses sharding manager
        flops_counter: FLOPs counter instance
        lr_scheduler: learning rate scheduler
        config: actor config
        world_size: number of devices

    Returns:
        DataProto with metrics in meta_info
    """

    device = torch.cuda.current_device()

    # Ensure data is on CUDA before starting
    data = data.to('cuda')
    assert self._is_actor

    # Step 1: Load model & optimizer if FSDP offloading is enabled
    if offload_param:
        load_fsdp_param_and_grad(actor_module, device_id=device, load_grad=offload_grad)
    if offload_optimizer:
        load_fsdp_optimizer(actor_optimizer, device_id=device)

    # Ensure batch tensor is on CUDA
    data.batch = data.batch.cuda()

    log_gpu_memory_usage('Before update policy', logger=logger)

    with sharding_manager:
        # Step 2: preprocess sharded input for model
        data = sharding_manager.preprocess_data(data=data)

        # Step 3: Save slow (initial) weights
        slow_weights = [p.detach().clone().cpu() for p in actor_module.parameters() if p.requires_grad]

        # Step 4: Inner PPO updates (in-place, mutates actor_module)
        for _ in range(num_inner_steps):
            update_fn(data)

        # Step 5: Save fast (post-inner-loop) weights
        fast_weights = [p.detach().clone().cpu() for p in actor_module.parameters() if p.requires_grad]

        # Step 6: Compute lookahead direction and apply slow + α(fast - slow)
        with torch.no_grad():
            for p, w_slow, w_fast in zip(actor_module.parameters(), slow_weights, fast_weights):
                if p.requires_grad:
                    direction = w_fast - w_slow
                    new_weight = w_slow + step_size * direction
                    p.data.copy_(new_weight.to(p.device))

        # Step 7: Disable grad & zero .grad so that update_fn won't update model
        requires_grad_flags = []
        for p in actor_module.parameters():
            requires_grad_flags.append(p.requires_grad)
            p.requires_grad_(False)
            if p.grad is not None:
                p.grad.zero_()

        # Step 8: Final forward-only PPO step to collect metrics (NO weight update)
        with torch.no_grad():
            with Timer(name='update_policy', logger=None) as timer:
                metrics = update_fn(data)  # do not modify model

        # Step 9: Restore grad flags for next outer PPO step
        for p, flag in zip(actor_module.parameters(), requires_grad_flags):
            p.requires_grad_(flag)

        # Step 10: Logging
        delta_time = timer.last
        global_num_tokens = data.meta_info['global_token_num']
        estimated_flops, promised_flops = flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics['mfu/actor'] = estimated_flops * config.ppo_epochs / promised_flops / world_size

        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        metrics['actor/lr'] = lr

        log_gpu_memory_usage('After update policy', logger=logger)

        # Step 11: wrap into DataProto
        output = DataProto(meta_info={'metrics': metrics})
        output = sharding_manager.postprocess_data(data=output)
        output = output.to('cpu')

    # Step 12: Optional FSDP offload after training step
    if offload_param:
        offload_fsdp_param_and_grad(actor_module, offload_grad=offload_grad)
    if offload_optimizer:
        offload_fsdp_optimizer(actor_optimizer)
    torch.cuda.empty_cache()

    return output
