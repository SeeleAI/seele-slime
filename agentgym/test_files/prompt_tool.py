import re
import html
from typing import List, Dict, Any
from jinja2 import Template
from agentgym.test_files.models import Tool, ToolParameter, ToolCall, Messages

class PromptTool:

    @staticmethod
    def parse_tool_calls(content: str) -> List[ToolCall]:
        tool_calls = []
        xml_regex = r'<tool>(.*?)</tool>'
        
        for xml_match in re.finditer(xml_regex, content, re.DOTALL):
            try:
                tool_content = xml_match.group(1)
                parsed_tool = {}
                
                tag_regex = r'<(\w+)>(.*?)</\1>'
                for tag_match in re.finditer(tag_regex, tool_content, re.DOTALL):
                    tag_name, tag_value = tag_match.groups()
                    parsed_tool[tag_name] = html.unescape(tag_value.strip())
                
                if 'name' in parsed_tool:
                    retry_count = None
                    if 'retry' in parsed_tool:
                        try:
                            retry_count = int(parsed_tool['retry'])
                        except ValueError:
                            retry_count = None

                    tool_call = ToolCall(
                        id=parsed_tool.get('id', ''),
                        name=parsed_tool['name'],
                        retry=retry_count,
                        **{k: v for k, v in parsed_tool.items() if k not in ['id', 'name', 'retry']}
                    )
                    tool_calls.append(tool_call)
                    
            except Exception as e:
                continue
                
        return tool_calls

    @staticmethod
    def extract_tool_calls_from_messages(messages: Messages) -> List[ToolCall]:
        all_tool_calls = []

        for message in messages:
            content = message.get('content', '')
            if content:
                tool_calls = PromptTool.parse_tool_calls(content)
                all_tool_calls.extend(tool_calls)

        return all_tool_calls

    @staticmethod
    def parse_tools(tool_list_data: Dict[str, Any]) -> List[Tool]:
        if not tool_list_data or 'tools' not in tool_list_data:
            return []
            
        tools_data = tool_list_data['tools']
        if not isinstance(tools_data, list):
            return []
            
        valid_tools = []
        for tool_data in tools_data:
            if PromptTool._is_valid_tool_definition(tool_data):
                valid_tools.append(PromptTool._clean_tool(tool_data))
                
        return valid_tools

    @staticmethod
    def construct_system_prompt(tool_list_data: Dict[str, Any]) -> str:
        SYSTEM_PROMPT_TEMPLATE = """<role>
You are an AI assistant with expertise in large language models (LLMs) and model training. You have knowledge of various LLM frameworks, training techniques, and best practices. You can help with model development, troubleshooting, and optimization tasks. All your tools are running in a real data environment, which is based on Linux. Provide clear and helpful guidance based on the available tools and information.
</role>

<tools>
    <native_tools>
    {% for tool in tools %}
        <tool>
            <name>{{ tool.name }}</name>
            <description>{{ tool.description }}</description>
            {% if tool.params %}
            <parameters>
            {% for param in tool.params %}
                <parameter>
                    <name>{{ param.name }}</name>
                    <type>{{ param.type }}</type>
                    <description>{{ param.description }}</description>
                    <required>{{ param.required|string|lower }}</required>
                </parameter>
            {% endfor %}
            </parameters>
            {% endif %}
        </tool>
    {% endfor %}
    </native_tools>
</tools>

<tool_use_instruction>
Embed tool calls in your message content using XML format. Each tool needs unique single-digit ID (0-9).
Use literal characters like < > & in commands, do NOT use HTML entities like &lt; &gt; &amp;

Examples:

1. "Let me check the current directory first:

<tool>
<id>0</id>
<name>BashTool</name>
<command>pwd</command>
</tool>"

2. "I'll read the config file and also check the process status:

<tool>
<id>0</id>
<name>ReadFileTool</name>
<file_path>/etc/config.json</file_path>
</tool>

<tool>
<id>1</id>
<name>BashTool</name>
<command>ps aux | grep nginx</command>
</tool>"

3. "The previous command failed, let me retry with different parameters:

<tool>
<id>0</id>
<name>BashTool</name>
<retry>1</retry>
<command>ls -la /home</command>
</tool>"
</tool_use_instruction>"""
        try:
            tools = PromptTool.parse_tools(tool_list_data)
            template = Template(SYSTEM_PROMPT_TEMPLATE)
            return template.render(tools=tools)
        except Exception:
            template = Template(SYSTEM_PROMPT_TEMPLATE)
            return template.render(tools=[])

    @staticmethod
    def validate_tool(input_params: Dict[str, Any], tool_definition: Tool) -> Dict[str, Any]:
        errors = []
        converted_params = {}

        if not isinstance(input_params, dict):
            errors.append('input must be an object')
            return {'valid': False, 'errors': errors, 'converted_params': {}}

        for param_name, param_value in input_params.items():
            param_def = next((p for p in tool_definition.params if p.name == param_name), None)
            if param_def and param_value is not None:
                try:
                    expected_type = param_def.type.lower()
                    if expected_type in ['integer', 'int']:
                        converted_params[param_name] = int(param_value) if isinstance(param_value, str) else param_value
                    elif expected_type == 'number':
                        converted_params[param_name] = float(param_value) if isinstance(param_value, str) else param_value
                    elif expected_type in ['object', 'dict']:
                        if isinstance(param_value, str):
                            import json
                            converted_params[param_name] = json.loads(param_value)
                        else:
                            converted_params[param_name] = param_value
                    elif expected_type in ['array', 'list']:
                        if isinstance(param_value, str):
                            import json
                            converted_params[param_name] = json.loads(param_value)
                        else:
                            converted_params[param_name] = param_value
                    elif expected_type in ['boolean', 'bool']:
                        if isinstance(param_value, str):
                            converted_params[param_name] = param_value.lower() == 'true'
                        else:
                            converted_params[param_name] = bool(param_value)
                    else:
                        converted_params[param_name] = param_value
                except (ValueError, json.JSONDecodeError) as e:
                    errors.append(f"parameter '{param_name}' expected type '{param_def.type}' but got invalid value: {str(e)}")
                    converted_params[param_name] = param_value
            else:
                converted_params[param_name] = param_value

        for param in tool_definition.params:
            if not param.required:
                continue

            if param.name not in converted_params:
                errors.append(f"required parameter '{param.name}' is missing")
                continue

            if converted_params[param.name] is None:
                errors.append(f"required parameter '{param.name}' cannot be null")

        for param in tool_definition.params:
            if param.name not in converted_params or converted_params[param.name] is None:
                continue

            actual_type = PromptTool._get_actual_type(converted_params[param.name])
            expected_type = param.type.lower()

            if not PromptTool._is_type_compatible(actual_type, expected_type):
                errors.append(f"parameter '{param.name}' expected type '{param.type}' but got '{actual_type}'")

        for key in converted_params:
            param_def = next((p for p in tool_definition.params if p.name == key), None)
            if not param_def:
                errors.append(f"unknown parameter '{key}' not defined in tool specification")

        return {'valid': len(errors) == 0, 'errors': errors, 'converted_params': converted_params}

    @staticmethod
    def _is_valid_tool_definition(tool_data: Any) -> bool:
        return tool_data and isinstance(tool_data, dict) and tool_data.get('name') and tool_data.get('description')

    @staticmethod
    def _clean_tool(tool_data: Dict[str, Any]) -> Tool:
        params = []
        if 'params' in tool_data and isinstance(tool_data['params'], list):
            for param_data in tool_data['params']:
                if isinstance(param_data, dict):
                    params.append(ToolParameter(
                        name=PromptTool._clean_text(param_data.get('name', '')),
                        type=PromptTool._clean_text(param_data.get('type', '')),
                        description=PromptTool._clean_text(param_data.get('description', '')),
                        required=bool(param_data.get('required', False))
                    ))
        
        return Tool(
            name=PromptTool._clean_text(tool_data['name']),
            description=PromptTool._clean_text(tool_data['description']),
            params=params
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ''
        return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    @staticmethod
    def _get_actual_type(value: Any) -> str:
        if isinstance(value, list):
            return 'array'
        elif value is None:
            return 'null'
        else:
            return type(value).__name__

    @staticmethod
    def _is_type_compatible(actual_type: str, expected_type: str) -> bool:
        if actual_type == expected_type:
            return True
            
        type_mapping = {
            'number': ['int', 'float'],
            'string': ['str'],
            'boolean': ['bool'],
            'object': ['dict'],
            'array': ['list']
        }
        
        return actual_type in type_mapping.get(expected_type, [])
