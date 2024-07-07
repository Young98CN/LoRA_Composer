import os

link_mapping = {
    "potter+hermione+rock_background": ["potter","hermione","rock"],
    "potter+hermione+pyramid_background+shake_hands": ["potter","hermione","pyramid"],
    "2real": ["potter","hermione"],
    "2real2": ["hinton","lecun"],
    "2real3": ["bengio","lecun"],
    "2dog+1cat": ["dogA","dogB","catA"],
    "5real": ["rock","potter","hermione","dogA","catA"],
    "5real2": ["rock","bengio","hinton","dogA","catA"],
    "5real3": ["rock","potter","hinton","dogA","catA"],
    "3real": ["lecun","bengio","hinton"],
    "4real": ["bengio","potter","hermione","dogA"],
    "4real2": ["hinton","potter","lecun","dogA"],
    "4real3": ["hinton","potter","hermione","catA"]
}

anime_mapping = {
    "5anime_2": ["hina_anythingv4","tezuka_anythingv4","ai_anythingv4","son_anythingv4","kario_anythingv4"],
    "5anime_3": ["hina_anythingv4","mitsuha_anythingv4","ai_anythingv4","son_anythingv4","kario_anythingv4"],
    "4anime": ["hina_anythingv4","mitsuha_anythingv4","ai_anythingv4","kario_anythingv4"],
    "4anime_2": ["ai_anythingv4","tezuka_anythingv4","son_anythingv4","kario_anythingv4"],
    "4anime_3": ["ai_anythingv4","hina_anythingv4","son_anythingv4","kario_anythingv4"],
    "3anime": ["ai_anythingv4","tezuka_anythingv4","kario_anythingv4"],
    "3girl": ["hina_anythingv4","mitsuha_anythingv4","kario_anythingv4"],
    "2boys": ["tezuka_anythingv4","son_anythingv4"],
}
script_directory = os.path.abspath(__file__).split("scripts")[0]
real_source = script_directory + 'experiments/composed_edlora/chilloutmix'
anime_source = script_directory + 'experiments/composed_edlora/anythingv4'
destination = 'experiments/link_folder'

for key, value in link_mapping.items():
    dst_folder = os.path.join(destination, key)
    os.makedirs(dst_folder, exist_ok=True)
    for i in value:
        source = os.path.join(real_source, i)
        dst = os.path.join(dst_folder, i)
        os.symlink(source, dst)
        
for key, value in anime_mapping.items():
    dst_folder = os.path.join(destination, key)
    os.makedirs(dst_folder, exist_ok=True)
    for i in value:
        source = os.path.join(anime_source, i)
        dst = os.path.join(dst_folder, i)
        os.symlink(source, dst)