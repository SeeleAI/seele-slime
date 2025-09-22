from dataclasses import dataclass, field, asdict
import json
from typing import List, Union, Dict, Callable, Any

@dataclass
class BaseToolSchema:
    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(asdict(self))

    def keys(self):
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()

    def items(self):
        return asdict(self).items()

@dataclass
class TestMathToolsSchema(BaseToolSchema):
    star_operation: dict = field(default_factory=lambda: {
        "type": "function",
        "function": {
            "name": "star_operation",
            "description": "Star Operation(☆), takes two integers as input and return the operation result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first operation number"},
                    "b": {"type": "integer", "description": "The second operation number"}
                },
                "required": ["a", "b"]
            }
        }
    })
    flower_operation: dict = field(default_factory=lambda: {
        "type": "function",
        "function": {
            "name": "flower_operation",
            "description": "Flower Operation(❀), takes two intergers as input and return the operation result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first operation number"},
                    "b": {"type": "integer", "description": "The second operation number"}
                },
                "required": ["a", "b"]
            }
        }
    })
    moon_operation: dict = field(default_factory=lambda: {
        "type": "function",
        "function": {
            "name": "moon_operation",
            "description": "Moon operation(☽), takes two integers as input and return the operation result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first operation number"},
                    "b": {"type": "integer", "description": "The second operation number"}
                },
                "required": ["a", "b"]
            }
        }
    })
    sun_operation: dict = field(default_factory=lambda: {
        "type": "function",
        "function": {
            "name": "sun_operation",
            "description": "Sun Operation(☀), takes two integers as input and return the operation result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first operation number"},
                    "b": {"type": "integer", "description": "The second operation number"}
                },
                "required": ["a", "b"]
            }
        }
    })
    
@dataclass
class MemoryToolsSchema(BaseToolSchema):
    summarize: dict = field(default_factory=lambda: {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": (
                "When you call this tool, you must provide a summarized text as argument, This tool will "
                "flush out all conversation history except the system prompt, use carefully and make sure your "
                "summarization is adaquate, but also make sure the summarization is brief so it won't exceed your "
                "context length."
                ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "str", "description": "Your summarization of the conversation history"}
                },
                "required": ["text"]
            }
        }
    })
    
def register_tool_schema(
    names: Union[str, Dict[str, List[str]]],
    *schemas: BaseToolSchema
) -> List[Dict]:
    """
    Retrieves tool schemas based on the provided names and schema instances.

    Args:
        names: Either the string 'all' to get all tools, or a dictionary
               specifying which tools to get from which schema classes,
               e.g., {'TestMathToolsSchema': ['sun_operation', 'moon_operation']}.
        *schemas: A variable number of schema instances (e.g., TestMathToolsSchema()).

    Returns:
        A flat list of the requested tool schema dictionaries.
        
    Raises:
        TypeError: If 'names' is not the string 'all' or a dictionary.
        KeyError: If a class name in the dictionary does not match any provided schema.
    """
    registered_tools = []

    if names == 'all':
        for schema in schemas:
            registered_tools.extend(schema.values())
    elif isinstance(names, dict):
        # Create a map of {class_name_string: schema_instance} for easy lookup
        schema_map = {schema.__class__.__name__: schema for schema in schemas}

        for class_name, tool_list in names.items():
            schema_instance = schema_map.get(class_name)
            if not schema_instance:
                raise KeyError(f"Schema class '{class_name}' not found in provided schemas.")
            
            for tool_name in tool_list:
                try:
                    # Use the __getitem__ method from BaseToolSchema
                    tool_data = schema_instance[tool_name]
                    registered_tools.append(tool_data)
                except AttributeError:
                    # This will catch cases where the tool_name is not an attribute
                    print(f"Warning: Tool '{tool_name}' not found in schema '{class_name}'.")
    else:
        raise TypeError("The 'names' argument must be 'all' or a dictionary.")
        
    return registered_tools


class UnifiedToolExecutor:
    """
    A unified class to execute various tool functions from a single interface.
    
    This class is not meant to be instantiated directly but rather by the
    `register_tool_executor` factory function.
    """
    def __init__(self, tool_map: Dict[str, Callable]):
        """
        Initializes the executor with a map of tool names to their functions.

        Args:
            tool_map (Dict[str, Callable]): A dictionary where keys are tool names
                                            and values are the callable functions.
        """
        self._tool_map = tool_map

    def forward(self, name: str, **kwargs) -> Any:
        """
        Calls a tool function by its name with the given arguments.

        Args:
            name (str): The name of the function to call (e.g., 'star_operation').
            **kwargs: The arguments to pass to the function (e.g., a=5, b=3).

        Returns:
            The result returned by the called tool function.
            
        Raises:
            ValueError: If the tool name is not found in the map.
        """
        if name not in self._tool_map:
            available = self.list_tools()
            raise ValueError(f"Tool '{name}' not found. Available tools are: {available}")
        
        # Retrieve the function from the map and call it with the provided arguments
        tool_function = self._tool_map[name]
        return tool_function(**kwargs)

    def list_tools(self) -> list[str]:
        """Returns a list of all available tool names."""
        return list(self._tool_map.keys())

def register_tool_executor(*tool_instances: Any) -> UnifiedToolExecutor:
    """
    Creates a UnifiedToolExecutor from one or more tool class instances.

    This function inspects the provided tool instances, collects their available
    methods based on the `available_tools` attribute, and registers them into a
    single executor object with a unified `forward` method.

    Args:
        *tool_instances: A variable number of instantiated tool classes
                         (e.g., TestMathTools(avaialble_tools='all')).

    Returns:
        An instance of the UnifiedToolExecutor class, ready to call any registered tool.
    """
    tool_map = {}
    print("Registering tools...")

    for instance in tool_instances:
        # Ensure the instance is properly configured to list its tools
        if not hasattr(instance, 'available_tools'):
            print(f"Warning: Instance {instance.__class__.__name__} has no 'available_tools' attribute. Skipping.")
            continue
        
        for tool_name in instance.available_tools:
            if tool_name in tool_map:
                # Handle cases where two classes have a tool with the same name
                print(f"Warning: Duplicate tool name '{tool_name}'. Overwriting with the one from {instance.__class__.__name__}.")
            
            # Get the actual method from the instance object
            tool_method = getattr(instance, tool_name)
            tool_map[tool_name] = tool_method
            print(f"  - Registered '{tool_name}' from {instance.__class__.__name__}")
    
    print("Registration complete!")
    return UnifiedToolExecutor(tool_map)


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
    # tools = register_tool_schema({"TestMathToolsSchema": ["flower_operation", "sun_operation"]}, TestMathToolsSchema(), MemoryToolsSchema())
    # print(json.dumps(tools, indent=2, ensure_ascii=False))
    tools = register_tool_schema("all", TestMathToolsSchema(), MemoryToolsSchema())
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    
    from agent.tools.test_math_tools import TestMathTools
    from agent.tools.memory_tools import MemoryTools
    exectutor = register_tool_executor(TestMathTools(avaialble_tools="all"), MemoryTools(avaialble_tools="all"))
    print(exectutor.forward("flower_operation", a=5, b=3))
    print(exectutor.forward("summarize", text="I am a test.", messages=[{"role": "system", "content": "text"}, {"role": "user", "content": "text"}, {"role": "assistant", "content": "123123"}]))
