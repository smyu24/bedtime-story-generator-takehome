import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages loading and formatting prompts from external files"""
    
    def __init__(self, prompt_file: str = "prompts.yaml"):
        """
        Initialize prompt manager
        
        Args:
            prompt_file: Path to YAML or JSON file containing prompts
        """
        self.prompt_file = Path(prompt_file)
        self.prompts: Dict[str, Any] = {}
        self.load_prompts()
    
    def load_prompts(self) -> None:
        """Load prompts from file"""
        if not self.prompt_file.exists():
            logger.warning(f"Prompt file {self.prompt_file} not found. Using defaults.")
            self._load_defaults()
            return
        
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                if self.prompt_file.suffix in ['.yaml', '.yml']:
                    self.prompts = yaml.safe_load(f)
                elif self.prompt_file.suffix == '.json':
                    self.prompts = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {self.prompt_file.suffix}")
            
            logger.info(f"Loaded prompts from {self.prompt_file}")
            logger.info(f"Prompt versions: {self._get_versions()}")
        
        except Exception as e:
            logger.error(f"Error loading prompts: {e}. Using defaults.")
            self._load_defaults()
    
    def _get_versions(self) -> Dict[str, str]:
        """Get version numbers of all prompts"""
        versions = {}
        for key in ['categorization', 'generation', 'judge', 'revision']:
            if key in self.prompts:
                versions[key] = self.prompts[key].get('version', 'unknown')
        return versions
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        Get a formatted prompt with variables substituted
        
        Args:
            prompt_type: Type of prompt (categorization, generation, judge, revision)
            **kwargs: Variables to substitute in the prompt template
        
        Returns:
            Formatted prompt string
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        template = self.prompts[prompt_type].get('user_template', '')
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable {e} for prompt {prompt_type}")
            raise
    
    def get_system_role(self, prompt_type: str) -> Optional[str]:
        """Get system role for a prompt type"""
        if prompt_type in self.prompts:
            return self.prompts[prompt_type].get('system_role')
        return None
    
    def get_temperature(self, prompt_type: str) -> float:
        """Get temperature setting for a prompt type"""
        if prompt_type in self.prompts:
            return self.prompts[prompt_type].get('temperature', 0.1)
        return 0.1
    
    def get_max_tokens(self, prompt_type: str) -> int:
        """Get max tokens setting for a prompt type"""
        if prompt_type in self.prompts:
            return self.prompts[prompt_type].get('max_tokens', 1000)
        return 1000
    
    def get_story_arc(self, category: str) -> str:
        """Get story arc template for a category"""
        arcs = self.prompts.get('story_arcs', {})
        return arcs.get(category, arcs.get('everyday', ''))
    
    def reload_prompts(self) -> None:
        """Reload prompts from file (useful for hot-reloading)"""
        logger.info("Reloading prompts...")
        self.load_prompts()
    
    def _load_defaults(self) -> None:
        """Load default prompts if file not found"""
        self.prompts = {
            'categorization': {
                'version': '1.0',
                'user_template': 'Categorize this story request: "{user_input}"',
                'temperature': 0.0,
                'max_tokens': 50
            },
            'generation': {
                'version': '1.0',
                'user_template': 'Write a story about: {user_input}',
                'temperature': 0.7,
                'max_tokens': 1500
            },
            'judge': {
                'version': '1.0',
                'user_template': 'Evaluate this story: {story}',
                'temperature': 0.2,
                'max_tokens': 800
            },
            'revision': {
                'version': '1.0',
                'user_template': 'Revise this story: {story}',
                'temperature': 0.7,
                'max_tokens': 1500
            },
            'story_arcs': {
                'everyday': 'Beginning → Middle → End'
            }
        }
    
    def export_prompts(self, output_file: str, format: str = 'yaml') -> None:
        """
        Export current prompts to a file
        
        Args:
            output_file: Path to output file
            format: 'yaml' or 'json'
        """
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format == 'yaml':
                    yaml.dump(self.prompts, f, default_flow_style=False, sort_keys=False)
                elif format == 'json':
                    json.dump(self.prompts, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported prompts to {output_path}")
        
        except Exception as e:
            logger.error(f"Error exporting prompts: {e}")
            raise