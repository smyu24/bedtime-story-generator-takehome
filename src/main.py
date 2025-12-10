import os
from openai import OpenAI
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from prompt_manager import PromptManager
from dotenv import load_dotenv

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

If I had 2 more hours, I would add a memory system to track previously generated stories to avoid repetition / build character consistency as well as enable multi-turn dialogue options such as allowing questions or request plot changes mid-story.
Additionally, having 
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("story_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StoryEvaluation(BaseModel):
    """Structured evaluation from the judge"""
    overall_score: float = Field(..., description="Overall quality score from 0-10")
    age_appropriateness: float
    engagement: float
    educational_value: float
    narrative_quality: float
    suggestions: List[str]
    approved: bool

class Story(BaseModel):
    """Story with metadata"""
    content: str
    category: str
    evaluation: Optional[StoryEvaluation] = None
    revision_count: int = 0

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class StoryGenerator:
    """Class to handle story generation"""
    def __init__(self, api_key: Optional[str] = None, prompt_file: str = "../prompts/v1.0/prompts.yaml"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found.")
        
        self.max_revision_attempts = 2

        # init prompt manager
        self.prompt_manager = PromptManager(prompt_file)
        self.age_min = 5
        self.age_max = 10
        self.min_words = 300
        self.max_words = 600
        self.approval_threshold_overall = 7.0
        self.approval_threshold_individual = 6.0

    def call_model(self, prompt: str, system_role: Optional[str] = None, 
                   max_tokens: int = 3000, temperature: float = 0.1) -> str:
        try:
            messages = []
            if system_role:
                messages.append({"role": "system", "content": system_role})
            messages.append({"role": "user", "content": prompt})

            resp = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature)

            return resp.choices[0].message.content
        
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise


    def categorize_request(self, user_input: str) -> str:
        """Classify the story request into categories for tailored generation"""
        prompt = self.prompt_manager.get_prompt('categorization', user_input=user_input)
        system_role = self.prompt_manager.get_system_role('categorization')
        temperature = self.prompt_manager.get_temperature('categorization')
        max_tokens = self.prompt_manager.get_max_tokens('categorization')

        try:
            category = self.call_model(
                prompt, 
                system_role=system_role,
                max_tokens=max_tokens, 
                temperature=temperature
            ).strip().lower()

            valid_categories = ['adventure', 'friendship', 'educational', 'fantasy', 'everyday', 'animal', 'mystery']
            return category if category in valid_categories else 'everyday'
        except Exception as e:
            logger.warning(f"Categorization failed: {e}, defaulting to 'everyday'")
            return 'everyday'

    def generate_initial_story(self, user_input: str, category: str) -> str:
        """Generate the first draft of the story"""
        arc_guidance = self.prompt_manager.get_story_arc(category)

        prompt = self.prompt_manager.get_prompt(
            'generation',
            user_input=user_input,
            category=category,
            arc_guidance=arc_guidance,
            age_min=self.age_min,
            age_max=self.age_max,
            min_words=self.min_words,
            max_words=self.max_words
        )

        system_role = self.prompt_manager.get_system_role('generation')
        temperature = self.prompt_manager.get_temperature('generation')
        max_tokens = self.prompt_manager.get_max_tokens('generation')

        return self.call_model(
            prompt, 
            system_role=system_role,
            max_tokens=max_tokens, 
            temperature=temperature
        )

    def judge_story(self, story: str, user_request: str) -> StoryEvaluation:
        """LLM judge evaluates story quality with structured criteria"""
        prompt = self.prompt_manager.get_prompt(
            'judge',
            user_request=user_request,
            story=story,
            age_min=self.age_min,
            age_max=self.age_max,
            approval_threshold_overall=self.approval_threshold_overall,
            approval_threshold_individual=self.approval_threshold_individual
        )

        system_role = self.prompt_manager.get_system_role('judge')
        temperature = self.prompt_manager.get_temperature('judge')
        max_tokens = self.prompt_manager.get_max_tokens('judge')

        try:
            response = self.call_model(
                prompt,
                system_role=system_role,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(response, prompt, system_role)

            eval_data = json.loads(response)

            return StoryEvaluation(
                overall_score=float(eval_data['overall_score']),
                age_appropriateness=float(eval_data['age_appropriateness']),
                engagement=float(eval_data['engagement']),
                educational_value=float(eval_data['educational_value']),
                narrative_quality=float(eval_data['narrative_quality']),
                suggestions=eval_data['suggestions'],
                approved=bool(eval_data['approved'])
            )
        
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")

            return StoryEvaluation(
                overall_score=5.0,
                age_appropriateness=5.0,
                engagement=5.0,
                educational_value=5.0,
                narrative_quality=5.0,
                suggestions=["Unable to evaluate - technical error"],
                approved=False
            )

    def revise_story(self, original_story: str, evaluation: StoryEvaluation, user_request: str) -> str:
        """Revise story based on judge feedback"""
        suggestions_text = "\n".join(f"- {s}" for s in evaluation.suggestions)

        prompt = self.prompt_manager.get_prompt(
            'revision',
            user_request=user_request,
            story=original_story,
            age_score=evaluation.age_appropriateness,
            engagement_score=evaluation.engagement,
            educational_score=evaluation.educational_value,
            narrative_score=evaluation.narrative_quality,
            overall_score=evaluation.overall_score,
            suggestions=suggestions_text,
            min_words=self.min_words,
            max_words=self.max_words
        )

        system_role = self.prompt_manager.get_system_role('revision')
        temperature = self.prompt_manager.get_temperature('revision')
        max_tokens = self.prompt_manager.get_max_tokens('revision')

        return self.call_model(
            prompt,
            system_role=system_role,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def generate_story_with_judge(self, user_input: str) -> Story:
        """Main pipeline: generate, judge, and revise if needed"""
        logger.info(f"Starting story generation for request: {user_input}")

        # categorize
        category = self.categorize_request(user_input)
        logger.info(f"Categorized as: {category}")

        # generate initial story
        story_text = self.generate_initial_story(user_input, category)
        story = Story(content=story_text, category=category)

        # Judge and iteratively improve
        for attempt in range(self.max_revision_attempts + 1):
            evaluation = self.judge_story(story.content, user_input)
            story.evaluation = evaluation

            logger.info(f"Attempt {attempt + 1} - Score: {evaluation.overall_score}/10 - Approved: {evaluation.approved}")

            if evaluation.approved or attempt == self.max_revision_attempts:
                break

            story.content = self.revise_story(story.content, evaluation, user_input)
            story.revision_count = attempt + 1

        logger.info(f"Final story generated with {story.revision_count} revisions")
        return story

    def interactive_session(self):
        """Interactive story generation with user feedback loop"""
        print("\n\nSTORY GENERATOR")
        print("-"*60)
        print("\nI'll create a wonderful bedtime story for ages 5-10!")
        print("After the story, you can request changes or hear a new story.\n")

        while True:
            user_input = input("What kind of story would you like? choose between ['adventure', 'friendship', 'educational', 'fantasy', 'everyday', 'animal', 'mystery'] (or 'quit' to exit): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                print("Please choose what kind of story you'd like\n")
                continue

            try:
                print("\nGenerating...\n")
                story = self.generate_story_with_judge(user_input)

                # display
                print("-"*60)
                print(f"STORY (Category: {story.category.title()})")
                print("-"*60)
                print(f"\n{story.content}\n")
                print("-"*60)

                if story.evaluation:
                    print(f"\nSTORY QUALITY REPORT:")
                    print(f"Overall Score: {story.evaluation.overall_score}/10")
                    print(f"Age Appropriateness: {story.evaluation.age_appropriateness}/10")
                    print(f"Engagement: {story.evaluation.engagement}/10")
                    print(f"Educational Value: {story.evaluation.educational_value}/10")
                    print(f"Narrative Quality: {story.evaluation.narrative_quality}/10")
                    print(f"Revisions Made: {story.revision_count}")
                    print(f"Status: {'✅ Approved' if story.evaluation.approved else '⚠️ Acceptable'}\n")

                # revisions
                revise = input("Would you like changes to this story? (yes/no): ").strip().lower()
                if revise in ['yes', 'y']:
                    change_request = input("What would you like changed? ").strip()
                    if change_request:
                        print("\n📖 Revising the story...\n")
                        revised_prompt = f"Original request: {user_input}\n\nChange requested: {change_request}"
                        story = self.generate_story_with_judge(revised_prompt)

                        print("-"*60)
                        print("Revisions Applied..")
                        print("-"*60)
                        print(f"\n{story.content}\n")
                        print("-"*60 + "\n")

            except Exception as e:
                logger.error(f"Error generating story: {e}")

def main():
    try:
        generator = StoryGenerator("../prompts/v1.0/prompts.yaml")
        generator.interactive_session()
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please set your OPENAI_API_KEY environment variable.\n")        
    except Exception as e:
        logger.error(f"{e}")

if __name__ == "__main__":
    main()