# 1. Installazione delle dipendenze in Colab
!pip install -q crewai crewai-tools

import os
from getpass import getpass
from crewai import Agent, Task, Crew, Process, LLM

# 2. Inserimento sicuro della chiave API di Groq
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass("Inserisci la tua GROQ API Key: ")

# 3. Configurazione del modello LLM tramite Groq
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.7
)

# 4. Definizione degli Agenti
ricercatore = Agent(
    role="Ricercatore di Tendenze Digitali",
    goal="Sintetizzare le novità principali su un argomento in 3 punti chiave",
    backstory="Sei un analista esperto nel sintetizzare concetti complessi per un pubblico scolastico.",
    verbose=True,
    llm=groq_llm
)

scrittore = Agent(
    role="Redattore Content Creator",
    goal="Creare un post coinvolgente per docenti basandosi sui punti chiave forniti",
    backstory="Sei un esperto di comunicazione didattica specializzato nella redazione di contenuti per insegnanti.",
    verbose=True,
    llm=groq_llm
)

# 5. Definizione dei Task
task_ricerca = Task(
    description="Analizza il tema 'L'uso degli Agenti IA nella didattica' e individua 3 opportunità chiave.",
    expected_output="Un elenco puntato con esattamente 3 opportunità analizzate.",
    agent=ricercatore
)

task_scrittura = Task(
    description="Prendi i 3 punti chiave della ricerca e scrivi un post divulgativo di 150 parole.",
    expected_output="Un post breve, chiaro e con tono professionale ed entusiasta.",
    agent=scrittore
)

# 6. Creazione della Crew ed Esecuzione
crew = Crew(
    agents=[ricercatore, scrittore],
    tasks=[task_ricerca, task_scrittura],
    process=Process.sequential
)

print("\n--- AVVIO LAVORO DEGLI AGENTI ---\n")
risultato = crew.kickoff()

print("\n--- RISULTATO FINALE ---\n")
print(risultato)
