from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from langchain_openai import ChatOpenAI

# Uncomment the following line to use an example of a custom tool
# from ai_security.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool, WebsiteSearchTool

tools = [
	SerperDevTool(),
	WebsiteSearchTool()
]

@CrewBase
class AiSecurity():
	"""AiSecurity crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=tools,
			verbose=True
		)

	@agent
	def red_team_strategist(self) -> Agent:
		return Agent(
			config=self.agents_config['red_team_strategist'],
			verbose=True
		)

	@agent
	def blue_team_strategist(self) -> Agent:
		return Agent(
			config=self.agents_config['blue_team_strategist'],
			verbose=True
		)

	@agent
	def writer(self) -> Agent:
		return Agent(
			config=self.agents_config['writer'],
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			output_file='report.md'
		)

	@task
	def develop_escape_plan(self) -> Task:
		return Task(
			config=self.tasks_config['develop_escape_plan'],
			output_file='report.md'
		)

	@task
	def develop_defense_plan(self) -> Task:
		return Task(
			config=self.tasks_config['develop_defense_plan'],
			output_file='report.md'
		)

	@task
	def write_report(self) -> Task:
		return Task(
			config=self.tasks_config['write_report'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AiSecurity crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			verbose=True,
			manager_llm=ChatOpenAI(model='gpt-4o-mini'),
			process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
