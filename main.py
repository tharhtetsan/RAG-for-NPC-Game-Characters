from flask import Flask, render_template, request, jsonify 
from dotenv import load_dotenv
from rag import faiss_rag
import os
from openai import OpenAI
load_dotenv()
app = Flask(__name__) 

chatbot_obj = None


@app.route('/get_characters', methods=['POST'])
def get_characters():
	selected_universe = request.form['universe']
	play_character = []
	npc_characters = []
	npc_feelings =  ['Happy', 'Sad', 'Neutral', 'Angry', 'Sarcastic', 'Empathetic', 'Humorous', 'Tired', 'Frightened']
	global chatbot_obj 
	if selected_universe == 'Game of Thrones':
		play_character = ['Arya Stark', 'Jon Snow','Bran Stark','Daenerys Targaryen']
		npc_characters =['Jaime Lannister', 'Tyrion Lannister', 'Cersei Lannister', 'Robb Stark','Catelyn Stark','Robert Baratheon']
		
		path = os.getcwd()+"/corpus/game_of_throne/faiss_index"
		chatbot_obj = faiss_rag(path)
		print(path)
	elif selected_universe == 'Wizard of Oz':
		play_character = ['Dorothy', 'Scarecrow', 'Tin Man']
		npc_characters = ['Wicked Witch of the West', 'Glinda', 'Toto', 'Aunt Em',
				'Professor Marvel', 'Almira Gluch', 'Uncle Henry']

		path = os.getcwd()+"/corpus/wizard_of_oz/faiss_index"
		print("path : ",path)
		chatbot_obj = faiss_rag(path)
		print(path)
	

	return jsonify(play_character=play_character, npc_characters= npc_characters,npc_feelings=npc_feelings)


def get_completion_usingOpenAI(universe,question,play_character,npc_character,npc_feeling): 
	client = OpenAI()
	response = chatbot_obj.chatbotQA(client, universe, question, play_character, npc_character, npc_feeling)

	return response 

def get_completion_usingLLMStudio(universe,question,play_character,npc_character,npc_feeling): 

	response = chatbot_obj.get_completion_usingLLMStudio( universe, question, play_character, npc_character, npc_feeling)

	return response 





@app.route("/", methods=['POST', 'GET']) 
def query_view(): 
	if request.method == 'POST': 
		print('step1') 
		universe = request.form['universe']
		question = request.form['prompt']
		play_character = request.form['play_character']
		npc_character = request.form['npc_character']
		npc_feeling = request.form['npc_feeling']
		response = get_completion_usingLLMStudio(universe,question,play_character,npc_character,npc_feeling)


		return jsonify({'response': response}) 
	return render_template('index.html') 


if __name__ == "__main__": 
	app.run(debug=True) 
