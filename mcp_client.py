#mcp_client1.py
#!/usr/bin/env python3
"""
Enhanced NLP MCP Client with Conversational Interface
Fixed MCP library usage and continuous conversational capability
"""

import asyncio
import json
import sys
import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configuration class
class Config:
    def __init__(self):
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.MCP_SERVER_SCRIPT = os.getenv('MCP_SERVER_SCRIPT', 'mcp_server.py')
        self.MCP_SERVER_TIMEOUT = int(os.getenv('MCP_SERVER_TIMEOUT', '30'))
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './client_output'))
        self.TEMP_DIR = Path(os.getenv('TEMP_DIR', './client_temp'))
        self.AUTO_EXECUTE_TOOLS = os.getenv('AUTO_EXECUTE_TOOLS', 'false').lower() == 'true'
        self.CONFIRM_BEFORE_EXECUTION = os.getenv('CONFIRM_BEFORE_EXECUTION', 'true').lower() == 'true'
        self.SAVE_CONVERSATION_HISTORY = os.getenv('SAVE_CONVERSATION_HISTORY', 'true').lower() == 'true'
        
        # Create directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Validate required settings
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        if not Path(self.MCP_SERVER_SCRIPT).exists():
            raise FileNotFoundError(f"MCP server script not found: {self.MCP_SERVER_SCRIPT}")

config = Config()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConversationalGemini:
    """Handles conversational interaction with Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.conversation_history = []
        self._configure_gemini()
    
    def _configure_gemini(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini API configured with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise
    
    async def analyze_user_intent(self, user_input: str, available_tools: List[Dict]) -> Dict[str, Any]:
        """Analyze user intent and determine if MCP tools are needed"""
        try:
            tools_summary = []
            for tool in available_tools:
                tools_summary.append({
                    "name": tool.get("name"),
                    "description": tool.get("description", "")[:100],
                    "category": self._categorize_tool(tool.get("name", ""))
                })
            
            analysis_prompt = f"""
You are an AI assistant that helps users by either having a conversation OR using specialized MCP tools for file processing tasks.

USER INPUT: "{user_input}"

AVAILABLE MCP TOOLS:
{json.dumps(tools_summary, indent=2)}

Analyze the user input and respond with JSON:
{{
    "intent_type": "conversation" or "tool_usage",
    "confidence": 0.0-1.0,
    "reasoning": "explain your analysis",
    "conversation_response": "if intent_type is conversation, provide a helpful response here",
    "tool_analysis": {{
        "needs_tools": true/false,
        "recommended_tools": ["tool1", "tool2"],
        "missing_info": ["what info is needed"],
        "file_paths_needed": ["expected file patterns"]
    }}
}}

RULES:
1. Use "conversation" for: greetings, questions, general chat, explanations, help requests
2. Use "tool_usage" for: file conversions, image processing, email analysis, document operations
3. High confidence (0.8+) for clear tool requests with file mentions
4. Medium confidence (0.5-0.7) for vague tool requests
5. Low confidence (0.3-0.5) for ambiguous requests

Respond ONLY with the JSON, no additional text.
"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, analysis_prompt
            )
            
            return self._parse_intent_response(response.text)
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "intent_type": "conversation",
                "confidence": 0.5,
                "conversation_response": "I'm having trouble understanding your request. Could you please rephrase it?"
            }
    
    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tools for better analysis"""
        if "document" in tool_name or "convert" in tool_name:
            return "document_processing"
        elif "image" in tool_name or "process" in tool_name:
            return "image_processing"
        elif "email" in tool_name or "search" in tool_name:
            return "email_analysis"
        elif "news" in tool_name or "trending" in tool_name:
            return "news_monitoring"
        else:
            return "utility"
    
    def _parse_intent_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini intent analysis response"""
        try:
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
            
        except json.JSONDecodeError:
            return {
                "intent_type": "conversation",
                "confidence": 0.5,
                "conversation_response": "I'm not sure I understood that correctly. Could you clarify?"
            }
    
    async def generate_tool_execution_plan(self, user_input: str, available_tools: List[Dict]) -> Dict[str, Any]:
        """Generate detailed execution plan for tool usage"""
        try:
            detailed_tools = []
            for tool in available_tools:
                tool_info = {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("inputSchema", {}).get("properties", {}),
                    "required": tool.get("inputSchema", {}).get("required", [])
                }
                detailed_tools.append(tool_info)
            
            plan_prompt = f"""
Create a detailed execution plan for this user request using available MCP tools.

USER REQUEST: "{user_input}"

AVAILABLE TOOLS:
{json.dumps(detailed_tools, indent=2)}

Create a JSON response with:
{{
    "can_execute": true/false,
    "confidence": 0.0-1.0,
    "execution_steps": [
        {{
            "step": 1,
            "tool_name": "exact_tool_name",
            "parameters": {{"param": "value"}},
            "description": "what this step does",
            "requires_user_input": true/false,
            "input_needed": ["list of inputs needed from user"]
        }}
    ],
    "expected_outcome": "what will be accomplished",
    "user_guidance": "instructions for the user"
}}

PARAMETER HANDLING:
- Use actual file paths if mentioned in the request
- Use placeholders like "USER_FILE_PATH" if file not specified
- Include all required parameters
- Set realistic default values where appropriate

Respond ONLY with JSON.
"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, plan_prompt
            )
            
            return self._parse_plan_response(response.text)
            
        except Exception as e:
            logger.error(f"Execution plan generation failed: {e}")
            return {"can_execute": False, "error": str(e)}
    
    def _parse_plan_response(self, response_text: str) -> Dict[str, Any]:
        """Parse execution plan response"""
        try:
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            return {
                "can_execute": False,
                "error": f"Failed to parse execution plan: {e}"
            }
    
    async def have_conversation(self, user_input: str, context: Dict = None) -> str:
        """Have a natural conversation with the user"""
        try:
            # Build context from conversation history
            conversation_context = ""
            if self.conversation_history:
                recent_history = self.conversation_history[-3:]  # Last 3 exchanges
                for entry in recent_history:
                    conversation_context += f"User: {entry.get('user', '')}\nAssistant: {entry.get('assistant', '')}\n"
            
            conversation_prompt = f"""
You are a helpful AI assistant integrated with an MCP server that provides document, image, and email processing capabilities.

Previous conversation context:
{conversation_context}

Current user input: "{user_input}"

Additional context: {json.dumps(context or {}, indent=2)}

Provide a helpful, natural, conversational response. Be friendly, informative, and engaging. If the user seems to need file processing help, gently guide them toward specific requests.

Keep responses concise but helpful.
"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, conversation_prompt
            )
            
            # Add to conversation history
            self.conversation_history.append({
                "user": user_input,
                "assistant": response.text.strip(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Conversation generation failed: {e}")
            return "I'm having trouble processing that right now. Could you try rephrasing your question?"

class SimpleMCPClient:
    """Simplified MCP client that works with direct communication"""
    
    def __init__(self):
        self.process = None
        self.available_tools = []
        self.connected = False
    
    async def connect(self):
        """Connect to MCP server using direct stdio communication"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, str(config.MCP_SERVER_SCRIPT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            logger.info(f"MCP server process started: {config.MCP_SERVER_SCRIPT}")
            
            # Initialize the server
            init_success = await self._initialize_server()
            if init_success:
                await self._load_tools()
                self.connected = True
                logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")
            else:
                raise Exception("Failed to initialize MCP server")
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.disconnect()
            raise
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.connected = False
        logger.info("Disconnected from MCP server")
    
    async def _initialize_server(self) -> bool:
        """Initialize the MCP server"""
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "clientInfo": {"name": "nlp-client", "version": "1.0.0"}
                }
            }
            
            await self._send_message(init_message)
            response = await self._receive_message()
            
            if response.get("error"):
                logger.error(f"Initialization failed: {response['error']}")
                return False
            
            # Send initialized notification
            initialized_msg = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            await self._send_message(initialized_msg)
            await asyncio.sleep(0.1)  # Give server time to process
            
            return True
            
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            return False
    
    async def _load_tools(self):
        """Load available tools from the server"""
        try:
            tools_message = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            await self._send_message(tools_message)
            response = await asyncio.wait_for(self._receive_message(), timeout=5.0)
            
            if response.get("error"):
                logger.error(f"Failed to load tools: {response['error']}")
                return
            
            result = response.get("result", {})
            tools = result.get("tools", [])
            
            self.available_tools = []
            for tool in tools:
                self.available_tools.append({
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "inputSchema": tool.get("inputSchema", {})
                })
            
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the MCP server"""
        try:
            tool_message = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            await self._send_message(tool_message)
            response = await asyncio.wait_for(self._receive_message(), timeout=30.0)
            
            if response.get("error"):
                return {
                    "success": False,
                    "error": response["error"].get("message", "Unknown error"),
                    "tool_name": tool_name
                }
            
            result = response.get("result", {})
            content = result.get("content", [])
            
            if content and len(content) > 0:
                content_text = content[0].get("text", "")
                try:
                    parsed_result = json.loads(content_text)
                    return {
                        "success": True,
                        "result": parsed_result,
                        "tool_name": tool_name
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "result": {"text": content_text},
                        "tool_name": tool_name
                    }
            
            return {
                "success": True,
                "result": {"message": "Tool executed successfully"},
                "tool_name": tool_name
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    async def _send_message(self, message: Dict):
        """Send JSON message to the server"""
        if not self.process:
            raise Exception("Not connected to server")
        
        json_str = json.dumps(message) + '\n'
        self.process.stdin.write(json_str.encode())
        await self.process.stdin.drain()
    
    async def _receive_message(self) -> Dict:
        """Receive JSON message from the server"""
        if not self.process:
            raise Exception("Not connected to server")
        
        line = await self.process.stdout.readline()
        if not line:
            raise Exception("No response from server")
        
        return json.loads(line.decode().strip())

class ConversationalMCPClient:
    """Main client that combines conversation and MCP tool execution"""
    
    def __init__(self):
        self.mcp_client = SimpleMCPClient()
        self.gemini = ConversationalGemini(config.GEMINI_API_KEY, config.GEMINI_MODEL)
        self.conversation_count = 0
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Connect to services"""
        await self.mcp_client.connect()
    
    async def disconnect(self):
        """Disconnect from services"""
        await self.mcp_client.disconnect()
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and return appropriate response"""
        try:
            self.conversation_count += 1
            
            # Analyze user intent
            intent_analysis = await self.gemini.analyze_user_intent(
                user_input, self.mcp_client.available_tools
            )
            
            intent_type = intent_analysis.get("intent_type", "conversation")
            confidence = intent_analysis.get("confidence", 0.5)
            
            if intent_type == "conversation" or confidence < 0.6:
                # Handle as conversation
                return await self.gemini.have_conversation(
                    user_input, {"available_tools": len(self.mcp_client.available_tools)}
                )
            
            else:
                # Handle as tool usage
                return await self._handle_tool_request(user_input, intent_analysis)
                
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
    
    async def _handle_tool_request(self, user_input: str, intent_analysis: Dict) -> str:
        """Handle requests that require MCP tool execution"""
        try:
            # Generate execution plan
            execution_plan = await self.gemini.generate_tool_execution_plan(
                user_input, self.mcp_client.available_tools
            )
            
            if not execution_plan.get("can_execute", False):
                return f"I understand you want to use tools, but I need more information: {execution_plan.get('user_guidance', 'Please provide more details about what you want to do.')}"
            
            # Execute the plan
            execution_results = []
            steps = execution_plan.get("execution_steps", [])
            
            for step in steps:
                tool_name = step.get("tool_name")
                parameters = step.get("parameters", {})
                
                # Check if we need user input
                if step.get("requires_user_input", False):
                    input_needed = step.get("input_needed", [])
                    return f"I can help with that! I need the following information to proceed:\n" + "\n".join(f"- {item}" for item in input_needed)
                
                # Replace placeholders if possible
                parameters = self._resolve_placeholders(parameters, user_input)
                
                # Execute the tool
                result = await self.mcp_client.execute_tool(tool_name, parameters)
                execution_results.append(result)
            
            # Generate response based on results
            return self._format_execution_response(execution_results, execution_plan)
            
        except Exception as e:
            logger.error(f"Tool request handling failed: {e}")
            return "I encountered an issue while trying to execute that request. Could you please try rephrasing it?"
    
    def _resolve_placeholders(self, parameters: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Resolve placeholder values in parameters"""
        resolved = parameters.copy()
        
        # Extract potential file paths
        file_patterns = [
            r'["\']([^"\']+\.[a-zA-Z0-9]{2,5})["\']',
            r'(\b\w+\.[a-zA-Z0-9]{2,5})\b'
        ]
        
        extracted_files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, user_input)
            extracted_files.extend(matches)
        
        # Replace placeholders
        for key, value in parameters.items():
            if isinstance(value, str) and ("USER_FILE_PATH" in value or "placeholder" in value.lower()):
                if extracted_files:
                    resolved[key] = extracted_files[0]
                else:
                    # Keep placeholder, will be handled by requiring user input
                    pass
        
        return resolved
    
    def _format_execution_response(self, execution_results: List[Dict], execution_plan: Dict) -> str:
        """Format response based on execution results"""
        successful_executions = [r for r in execution_results if r.get("success")]
        failed_executions = [r for r in execution_results if not r.get("success")]
        
        response_parts = []
        
        if successful_executions:
            response_parts.append(f"✓ Successfully executed {len(successful_executions)} operation(s)!")
            
            for result in successful_executions:
                tool_name = result.get("tool_name", "Unknown tool")
                response_parts.append(f"- {tool_name}: Completed successfully")
                
                # Include relevant result information
                if result.get("result") and isinstance(result["result"], dict):
                    if "output_path" in result["result"]:
                        response_parts.append(f"  Output saved to: {result['result']['output_path']}")
                    elif "message" in result["result"]:
                        response_parts.append(f"  {result['result']['message']}")
        
        if failed_executions:
            response_parts.append(f"\n✗ {len(failed_executions)} operation(s) failed:")
            for result in failed_executions:
                tool_name = result.get("tool_name", "Unknown tool")
                error = result.get("error", "Unknown error")
                response_parts.append(f"- {tool_name}: {error}")
        
        expected_outcome = execution_plan.get("expected_outcome", "")
        if expected_outcome and successful_executions:
            response_parts.append(f"\n{expected_outcome}")
        
        return "\n".join(response_parts) if response_parts else "Operation completed."
    
    async def start_interactive_session(self):
        """Start the interactive conversational session"""
        print("=" * 70)
        print("🤖 Conversational MCP Client - Interactive Session")
        print("=" * 70)
        print(f"Connected to: {config.MCP_SERVER_SCRIPT}")
        print(f"Available tools: {len(self.mcp_client.available_tools)}")
        print(f"AI Model: {config.GEMINI_MODEL}")
        print("\nI can help you with:")
        print("• Document conversion (PDF, DOCX, HTML, Markdown, etc.)")
        print("• Image processing (resize, enhance, format conversion)")
        print("• Email analysis and search")
        print("• News monitoring and trending topics")
        print("• General questions and conversation")
        print("\nJust tell me what you need in natural language!")
        print("Type 'quit' to exit.")
        print("=" * 70)
        
        while True:
            try:
                print(f"\n[{self.conversation_count + 1}]", end="")
                user_input = input(" You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🤖 Assistant: Goodbye! It was nice chatting with you!")
                    break
                
                # Show thinking indicator
                print("🤖 Assistant: ", end="", flush=True)
                
                # Process the input
                response = await self.process_user_input(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Assistant: Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"\n🤖 Assistant: I encountered an error: {e}")
                logger.error(f"Interactive session error: {e}")

async def main():
    """Main function"""
    try:
        print("🚀 Starting Enhanced Conversational MCP Client...")
        print(f"📁 Configuration loaded from .env file")
        print(f"📄 Server script: {config.MCP_SERVER_SCRIPT}")  
        print(f"🧠 AI Model: {config.GEMINI_MODEL}")
        
        async with ConversationalMCPClient() as client:
            if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
                # Single request mode
                user_request = " ".join(sys.argv[1:])
                print(f"\n💬 Processing: {user_request}")
                response = await client.process_user_input(user_request)
                print(f"🤖 Response: {response}")
            else:
                # Interactive mode
                await client.start_interactive_session()
                
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())