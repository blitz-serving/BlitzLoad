class PullModelRequest:
    def __init__(self, model_name, world_size):
        self.model_name = model_name
        self.world_size = world_size
class PullModelResponse:
    def __init__(self, task_id):
        self.task_id = task_id
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
    def __init__(self, handler, offset, loaded_size):
        self.handler = handler
        self.offset = offset
        self.loaded_size = loaded_size
class RevertHandlerRequest:
    def __init__(self, tensor_name, tensor_size, rank):
        self.tensor_name = tensor_name
        self.tensor_size = tensor_size
        self.rank = rank
class RevertHandlerResponse:
    def __init__(self, success):
        self.success = success
