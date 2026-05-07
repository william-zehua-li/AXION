import traceback


class ToolExecutor:
    def __init__(self, tools: dict):
        # tools: initial mapping of tool_name -> callable
        self._tools: dict = dict(tools)

    def register(self, name: str, fn):
        self._tools[name] = fn

    def execute(self, tool_name: str, args: dict):
        fn = self._tools.get(tool_name)
        if fn is None:
            return {
                "success": False,
                "error": f"Unknown tool: '{tool_name}'. "
                         f"Registered tools: {list(self._tools.keys())}",
                "output": None,
            }
        try:
            output = fn(**args) if isinstance(args, dict) else fn(args)
            return {"success": True, "output": output, "error": None}
        except Exception:
            return {
                "success": False,
                "error": traceback.format_exc(),
                "output": None,
            }
