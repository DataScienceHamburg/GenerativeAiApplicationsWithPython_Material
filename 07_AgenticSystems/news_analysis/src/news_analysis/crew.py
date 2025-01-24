#%% packages
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebsiteSearchTool
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%% Tools
search_tool = SerperDevTool()
website_search_tool = WebsiteSearchTool()

#%%
@CrewBase
class NewsAnalysis():
	"""NewsAnalysis crew"""

	tasks_config = 'config/tasks.yaml'
	agents_config = 'config/agents.yaml'

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=[search_tool, website_search_tool], # Example of custom tool, loaded on the beginning of file
			verbose=True
		)

	@agent
	def analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['analyst'],
			verbose=True
		)
  
	@agent
	def writer(self) -> Agent:
		return Agent(
			config=self.agents_config['writer'],
			verbose=True
		)

	@task
	def information_gathering_task(self) -> Task:
		return Task(
			config=self.tasks_config['information_gathering_task'],
		)

	@task
	def fact_checking_task(self) -> Task:
		return Task(
			config=self.tasks_config['fact_checking_task'],
		)

	@task
	def context_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['context_analysis_task'],
		)

	@task
	def report_assembly_task(self) -> Task:
		return Task(
			config=self.tasks_config['report_assembly_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the NewsAnalysis crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
		)
