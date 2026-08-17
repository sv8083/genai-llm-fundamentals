
from pathlib import Path
from typing import Dict


class PromptLoader:
    """Load evaluation prompts from files."""
    
    _prompts_cache: Dict[str, str] = {}
    _base_path: Path | None = Path(__file__).parent
        
    @classmethod
    def _ensure_base_path(cls):
        """Ensure base path is set."""
        if cls._base_path is None:
            cls.set_base_path()
    
    @classmethod
    def load(cls, prompt_name: str) -> str:
        """
        Load a prompt template from file.
        
        Args:
            prompt_name: Name of the prompt (e.g., 'rubric', 'sentiment_rubric')
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        cls._ensure_base_path()
        
        # Check cache first
        if prompt_name in cls._prompts_cache:
            return cls._prompts_cache[prompt_name]
        
        prompt_path = cls._base_path / f"{prompt_name}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r") as f:
            content = f.read()
        
        # Cache for future use
        cls._prompts_cache[prompt_name] = content
        return content
    
    @classmethod
    def render(cls, prompt_name: str, **kwargs) -> str:
        """
        Load and render a prompt template with variables.
        
        Args:
            prompt_name: Name of the prompt template
            **kwargs: Variables to format into the template
            
        Returns:
            Rendered prompt string
        """
        template = cls.load(prompt_name)
        return template.format(**kwargs)
