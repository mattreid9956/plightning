   
   
   @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=self.trainer.num_nodes * self.trainer.num_processes,
            rank=self.trainer.global_rank
        )
        if self.ddp_plugin is not None:
            distributed_sampler_kwargs = self.ddp_plugin.distributed_sampler_kwargs(distributed_sampler_kwargs)
        return distributed_sampler_kwargs
