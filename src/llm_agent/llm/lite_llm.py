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
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "timeout": self.timeout,
                "stream": False,
                "mock_response": "test" if response_format is None else None
            }

            # Add response format for models that support it
            if response_format and any(provider in self.model.lower() for provider in ["gpt", "claude", "anthropic"]):
                completion_kwargs["response_format"] = response_format

            #print(completion_kwargs)
            #input("waiting")

            response = await litellm.acompletion(**completion_kwargs)
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
        if "gpt" in self.model.lower():
            response_format = output_schema#{"type": "json_object"}
            '''
            if output_schema:
                messages[0]["content"] += f"\nPlease provide response matching this JSON schema: {json.dumps(output_schema)}"
            '''
        
        elif "claude" in self.model.lower():
            response_format = output_schema#{"type": "json"}
            '''
            if output_schema:
                messages[0]["content"] += f"\nPlease provide response matching this JSON schema: {json.dumps(output_schema)}"
            '''

        response = await self._generate_raw(messages, response_format)
        
        return response

        '''
        # Attempt to parse JSON response
        parsed_json = None
        if response_format:
            try:
                # Handle potential markdown code blocks in response
                if "```json" in generated_text:
                    generated_text = generated_text.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_text:
                    generated_text = generated_text.split("```")[1].strip()
                parsed_json = json.loads(generated_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {str(e)}")

        return GenerationResponse(
            text=generated_text,
            parsed_json=parsed_json,
            raw_response=raw_response.dict()
        )
        '''





