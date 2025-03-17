from tasks.ctx_dep_mante_task import CtxDepManteTask


__AVAILABLE_TASKS__ = {
    'ctx_dep_mante_task': CtxDepManteTask
}

def load_task(task_name: str, *args, **kwargs):
    if task_name not in __AVAILABLE_TASKS__:
        raise ValueError(f"Task {task_name} is not available. Available tasks are {list(__AVAILABLE_TASKS__.keys())}")
    
    return __AVAILABLE_TASKS__[task_name](*args, **kwargs)