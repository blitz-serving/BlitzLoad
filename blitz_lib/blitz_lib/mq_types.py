class PullModelRequest:
    def __init__(self, model_name, world_size, tp_size, pp_size):
        self.model_name = model_name
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
class PullModelResponse:
    def __init__(self, task_id):
        self.task_id = task_id
class PullDiffusionModelRequest:
    def __init__(self, file_name):
        self.file_name = file_name
class CheckModelRequest:
    def __init__(self, model_name, task_id):
        self.model_name = model_name
        self.task_id = task_id
class CheckModelResponse:
    def __init__(self, done):
        self.done = done
class LoadTensorRequest:
    def __init__(self, tensor_name, tensor_size, rank):
        self.tensor_name = tensor_name
        self.tensor_size = tensor_size
        self.rank = rank
class LoadTensorResponse:
    def __init__(self, handler, offset, loaded_size, resize_tensor):
        self.handler = handler
        self.offset = offset
        self.loaded_size = loaded_size
        self.resize_tensor = resize_tensor
class RevertHandlerRequest:
    def __init__(self, tensor_name, tensor_size, rank):
        self.tensor_name = tensor_name
        self.tensor_size = tensor_size
        self.rank = rank
class RevertHandlerResponse:
    def __init__(self, success):
        self.success = success
class GetMetaRequest:
    def __init__(self, file_name):
        self.file_name = file_name
class GetMetaResponse:
    def __init__(self, meta_str):
        self.meta_str = meta_str
class GetMetaTensorRequest:
    def __init__(self, file_name):
        self.file_name = file_name
class GetMetaTensorResponse:
    def __init__(self, meta_tensors):
        self.meta_tensors = meta_tensors
class ResetStatusRequest:
    def __init__(self, rank):
        self.rank = rank
class EmptyRequestResponse:
    def __init__(self):
        pass
