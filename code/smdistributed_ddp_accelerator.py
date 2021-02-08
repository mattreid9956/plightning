import torch
from pytorch_lightning.accelerators.ddp_accelerator import DDPAccelerator 
   
import smdistributed.dataparallel.torch.parallel.distributed as dist

dist.get_world_size()

class SagemakerDDPAccelerator(DDPAccelerator):
     
    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = dist.get_local_rank() # process_idx
        self.trainer.global_rank = dist.get_rank() # self.trainer.node_rank * self.trainer.num_processes + process_idx
        self.trainer.world_size = dist.get_world_size() # self.trainer.num_nodes * self.trainer.num_processes

#    def init_device(self, process_idx):
#         self.trainer.root_gpu = self.trainer.data_parallel_device_ids[self.trainer.local_rank]
#        torch.cuda.set_device(dist.get_local_rank())
#         torch.cuda.set_device(self.trainer.root_gpu)
      
    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank(),
        )
        if self.ddp_plugin is not None:
            distributed_sampler_kwargs = self.ddp_plugin.distributed_sampler_kwargs(distributed_sampler_kwargs)
        return distributed_sampler_kwargs
