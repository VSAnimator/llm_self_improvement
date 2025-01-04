import litellm
import json
from typing import Dict, List, Optional, Any, Union
from logging import getLogger
from pydantic import BaseModel

logger = getLogger(__name__)

class GenerationResponse(BaseModel):
    """Structured response from LLM generation"""
    text: str
    parsed_json: Optional[Dict[str, Any]] = None
    raw_response: Dict[str, Any]

class LiteLLMWrapper:
    def __init__(self, config: Dict):
        """Initialize LiteLLM wrapper with config parameters"""
        self.model = config['llm']['model']
        self.max_tokens = config['llm']['max_tokens']
        self.temperature = config['llm']['temperature']
        self.top_p = config['llm']['top_p']
        self.frequency_penalty = config['llm']['frequency_penalty']
        self.presence_penalty = config['llm']['presence_penalty']
        self.timeout = config['llm']['timeout_seconds']

    async def _generate_raw(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Base generation method"""
        try:
            completion_kwargs = {
                "model": self.model, #"openai/meta-llama/Llama-3.1-8B-Instruct",#self.model,
                "messages": messages,
                "max_tokens": 1024,#self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                #"frequency_penalty": self.frequency_penalty,
                #"presence_penalty": self.presence_penalty,
                "timeout": self.timeout,
                "stream": False,
                #"api_base": "http://0.0.0.0:8000/v1"
                #"mock_response": "test" if response_format is None else None
                #"drop_params": True, # In case the model doesn't have some of the parameters
                "stop": ["\n"]
            }

            # Add response format for models that support it
            if response_format: # and any(provider in self.model.lower() for provider in ["openai", "claude", "anthropic", "gemini", "together"]):
                completion_kwargs["response_format"] = response_format

            #print(messages)
            #input("Waiting")

            response = await litellm.acompletion(**completion_kwargs)

            #print(response)
            #input("Waiting")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def generate_text(self, prompt: str) -> str:
        """
        Generate free-form text response from a single prompt
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self._generate_raw(messages)
        return response.choices[0].message.content

    async def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate text response from a conversation history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Generated text response
        """
        response = await self._generate_raw(messages)
        return response.choices[0].message.content

    async def generate_structured(
        self, 
        messages: List[Dict[str, str]],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> GenerationResponse:
        """
        Generate structured response with full metadata
        
        Args:
            messages: Input conversation history
            output_schema: Optional schema specification for structured output
            
        Returns:
            GenerationResponse object containing generated text and raw response
        """
        response_format = None
        
        # Handle different model capabilities for structured output
        if "gpt" in self.model.lower() or "openai" in self.model.lower():
            response_format = output_schema

        elif "gemini" in self.model.lower() or "together" in self.model.lower():
            response_format = {"type": "json_object", "response_schema": output_schema.model_json_schema(), "enforce_validation": True}
        
        elif "claude" in self.model.lower():
            response_format = output_schema

        response = await self._generate_raw(messages, response_format)
        
        return response






