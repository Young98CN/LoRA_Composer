import os

link_mapping = {
    "potter+hermione+rock_background": ["potter_old.pth","hermione_old.pth","edlora_rock.pth"],
    "potter+hermione+pyramid_background+shake_hands": ["potter_old.pth","hermione_old.pth","edlora_pyramid.pth"],
    "2real": ["potter_old.pth","hermione_old.pth"],
    "2real2": ["hinton.pth","lecun.pth"],
    "2real3": ["bengio.pth","potter_old.pth"],
    "2dog+1cat": ["dogA.pth","dogB.pth","catA.pth"],
    "5real": ["edlora_rock.pth","potter_old.pth","hermione_old.pth","dogA.pth","catA.pth"],
    "5real2": ["edlora_rock.pth","bengio.pth","hinton.pth","dogA.pth","catA.pth"],
    "5real3": ["edlora_rock.pth","potter_old.pth","hinton.pth","dogA.pth","catA.pth"],
    "3real": ["lecun.pth","bengio.pth","hinton.pth"],
    "4real": ["hinton.pth","bengio.pth","hermione_old.pth","dogA.pth"],
    "4real2": ["hinton.pth","potter_old.pth","lecun.pth","dogA.pth"],
    "4real3": ["hinton.pth","potter_old.pth","hermione_old.pth","catA.pth"]
}

anime_mapping = {
    "5anime_2": ["edlora_hina.pth","tezuka_old.pth","ai.pth","edlora_son.pth","edlora_kaori.pth"],
    "5anime_3": ["edlora_hina.pth","edlora_mitsuha.pth","ai.pth","edlora_son.pth","edlora_kaori.pth"],
    "4anime": ["edlora_hina.pth","edlora_mitsuha.pth","ai.pth","edlora_kaori.pth"],
    "4anime_2": ["ai.pth","tezuka_old.pth","edlora_son.pth","edlora_kaori.pth"],
    "4anime_3": ["ai.pth","edlora_hina.pth","edlora_son.pth","edlora_kaori.pth"],
    "3anime": ["ai.pth","tezuka_old.pth","edlora_kaori.pth"],
    "3girl": ["edlora_hina.pth","edlora_mitsuha.pth","edlora_kaori.pth"],
    "2boys": ["tezuka_old.pth","edlora_son.pth"],
}
script_directory = os.path.abspath(__file__).split("scripts")[0]
real_source = script_directory + 'loras/real'
anime_source = script_directory + 'loras/anime'
bg_source = script_directory + 'loras/background'
destination = 'experiments/link_folder'

for key, value in link_mapping.items():
    dst_folder = os.path.join(destination, key)
    os.makedirs(dst_folder, exist_ok=True)
    for i in value:
        if i in ["edlora_rock.pth", "edlora_pyramid.pth"]:
            source = os.path.join(bg_source, i)
        else:
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