import litellm
import json
from typing import Dict, List, Optional, Any, Union
from logging import getLogger
from pydantic import BaseModel
from .sglang import SGLangBackend

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

        
        if config['llm']['backend'] == "litellm":
            self._generate_raw = self.__generate_raw
        elif config['llm']['backend'] == "sglang":
            self._generate_raw = self._generate_raw_sglang
        else:
            raise ValueError(f"Invalid backend: {config['llm']['backend']}")


    async def _generate_raw_sglang(
            self,
            messages: List[Dict[str, str]],
            response_format: Optional[Dict[str, str]] = None,
            stop: Optional[List[str]] = ["\n"],
        ) -> Dict[str, Any]:
            """Base generation method"""
            try:
                # Merge messages into a single user message
                user_message = "\n".join([f"{msg['content']}" for msg in messages])
                response = await SGLangBackend.generate(
                    messages=[{"role": "user", "content": user_message}],
                    max_tokens=self.max_tokens,
                    stop=stop,
                    response_format=response_format,
                )
                return response

            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise

    async def __generate_raw(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Optional[Dict[str, str]] = None,
        stop: Optional[List[str]] = ["\n"]
    ) -> Dict[str, Any]:
        """Base generation method"""
        try:
            if "human" in self.model.lower():
                print(messages)
            completion_kwargs = {
                "model": self.model, #"openai/meta-llama/Llama-3.1-8B-Instruct",#self.model,
                "messages": messages,
                "max_tokens": 1024,#self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                #"frequency_penalty": self.frequency_penalty,
                #"presence_penalty": self.presence_penalty,
                "timeout": 120.0,
                "stream": False,
                #"api_base": "http://0.0.0.0:8000/v1"
                "mock_response": input("Human: ") if "human" in self.model.lower() else None,
                #"drop_params": True, # In case the model doesn't have some of the parameters
                "stop": stop
            }

            # Add response format for models that support it
            if response_format: # and any(provider in self.model.lower() for provider in ["openai", "claude", "anthropic", "gemini", "together"]):
                completion_kwargs["response_format"] = response_format

            response = await litellm.acompletion(**completion_kwargs)

            return response.choices[0].message.content

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
        return response

    async def generate_chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = ["\n"]) -> str:
        """
        Generate text response from a conversation history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Generated text response
        """
        response = await self._generate_raw(messages, stop=stop)
        return response


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
        if "gemini" in self.model.lower() or "together" in self.model.lower():
            response_format = {"type": "json_object", "response_schema": output_schema.model_json_schema(), "enforce_validation": True}
        
        else:
            response_format = output_schema
        
        response = await self._generate_raw(messages, response_format)
        
        return response






