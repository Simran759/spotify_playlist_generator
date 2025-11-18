#!/usr/bin/env python3

import keyring 
import argparse
import spotipy
import subprocess
import webbrowser
import numpy as np
from scipy.spatial import distance # For comparing vectors

# --- NEW ---
# Import the Deep Learning model for NLP
from sentence_transformers import SentenceTransformer

# Parsing arguments for command line calls
parser = argparse.ArgumentParser(description='Simple command line song utility')
parser.add_argument("-p", type=str, help='The prompt to describe the playlist')
parser.add_argument("-l", type=int, default=10, help='The length of the playlist')
parser.add_argument("-n", type=str, default=argparse.SUPPRESS, help='The name of the playlist')
parser.add_argument("-i", action='store_true', default=False, help='Build interactive or automatic playlist')
args = parser.parse_args()
if 'n' not in args:
    args.n = args.p

separator = f'{100*"-"}'

class SpotifyPlaylist:

    def __init__(self, prompt, length=10, name=None, interactive=False):
        self.prompt = prompt
        self.length = int(length)
        self.name = name if name is not None else prompt
        self.interactive = interactive
        self.generated_tracks = [] 
        self.playlist_tracks = set()
        self.artists_blacklist = set()
        self.songs_blacklist = set()
        self.songs_in_playlist = set()
        self.sp = None
        self.current_user = None
        self.playlist = None
        
        # --- NEW ---
        # Load the pre-trained NLP model when the class is created.
        # This will download the model (about 200-300MB) the first time you run it.
        print("Loading deep learning NLP model (this may take a moment)...")
        # self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        print("Model loaded.")

    def __repr__(self):
        # ... (This function is unchanged) ...
        result = f'{separator}\n'
        result += f'Prompt: {self.prompt}\n'
        result += f'Length: {self.length}\n'
        result += f'Name: {self.name}\n'
        result += f'Interactive: {self.interactive}\n'
        if self.artists_blacklist:
            result += f'Artists Blacklist: {self.artists_blacklist}\n'
        if self.songs_blacklist:
            result += f'Songs Blacklist: {self.songs_blacklist}\n'
        result += f'{separator}\n'

        if not self.playlist:
            return result + "Playlist not created yet.\n"

        try:
            playlist_items = self.sp.playlist_items(self.playlist['id'])['items']
            for id, item in enumerate(playlist_items, start=1):
                track = item.get('track')
                if track:
                    artist = track['album']['artists'][0]['name']
                    name = track['name']
                    album = track['album']['name']
                    result += f"{id}. {artist} - {name} | {album}\n"
            result += f'{separator}\n'
        except Exception as e:
            result += f"Could not retrieve playlist items: {e}\n"
        
        return result

    # --- NEW ---
    # This function REPLACES all previous generation functions.
    def generate_nlp_playlist(self):
        """
        Generates a song list using an NLP model to find
        semantically similar tracks to the prompt.
        """
        print(f"Generating vector for prompt: '{self.prompt}'")
        
        # 1. Convert the user's text prompt into a vector "fingerprint"
        # The .encode() method runs the deep learning model
        prompt_vector = self.nlp_model.encode(self.prompt, convert_to_tensor=False)

        # 2. Get a large list of candidate tracks from Spotify
        print("Getting 50 candidate songs from Spotify search...")
        try:
            search_results = self.sp.search(q=self.prompt, type='track', limit=50)
            candidates = search_results['tracks']['items']
            if not candidates:
                print(f"Error: Could not find any tracks matching '{self.prompt}'. Try a different prompt.")
                return []
        except Exception as e:
            print(f"Error searching Spotify: {e}")
            return []

        # 3. Create a list of "text" for each song
        candidate_texts = []
        for track in candidates:
            # We create a "sentence" for each song
            text = f"{track['artists'][0]['name']} - {track['name']}"
            candidate_texts.append(text)

        # 4. Run the DL model on ALL 50 candidates at once
        print("Analyzing all 50 candidates with the NLP model...")
        candidate_vectors = self.nlp_model.encode(candidate_texts, convert_to_tensor=False)

        # 5. Calculate Similarity
        # We compare the single prompt_vector to all 50 candidate_vectors
        print("Calculating semantic similarity...")
        similarities = []
        for i, (track, vector) in enumerate(zip(candidates, candidate_vectors)):
            # Calculate Cosine Similarity (1.0 is identical, -1.0 is opposite)
            # (1 - distance) because distance.cosine returns 0.0 for identical
            sim = 1 - distance.cosine(prompt_vector, vector)
            
            # We store the similarity score and the full track object
            similarities.append((sim, track))
            # print(f"  - {candidate_texts[i]} (Similarity: {sim:.4f})") # Uncomment for debugging
        
        # 6. Sort by most similar and filter
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        final_tracks = []
        for sim, track in similarities:
            # Filter out blacklisted songs/artists
            if (track['name'] in self.songs_blacklist or 
                track['artists'][0]['name'] in self.artists_blacklist):
                continue
            
            final_tracks.append(track)
            if len(final_tracks) >= self.length:
                break # Stop once we have enough songs
                
        return final_tracks


    def login_to_spotify(self):
        # ... (This function is unchanged) ...
        client_id = keyring.get_password('spotify', 'client_ID')
        client_secret = keyring.get_password('spotify', 'client_secret')

        if not client_id or not client_secret:
            print("CRITICAL ERROR: Spotify Client ID or Secret not found.")
            exit()

        print("Client ID and Client Secret found.")
        
        self.sp = spotipy.Spotify(
            auth_manager=spotipy.SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri='http://127.0.0.1:9999',
                scope='user-read-playback-state,user-modify-playback-state,playlist-modify-private'
            )
        )
        self.current_user = self.sp.current_user()
        assert self.current_user is not None

    def main(self):
        # ... (This function is unchanged) ...
        self.login_to_spotify()

        all_playlists = self.sp.user_playlists(self.current_user['id'])
        playlist_names = [p['name'].lower() for p in all_playlists['items']]
        
        p_name = f'_{self.name}'
        playlist_name = p_name
        id = 1

        while playlist_name.lower() in playlist_names:
            id += 1
            playlist_name = f'{p_name} {str(id)}'
        
        self.playlist = self.sp.user_playlist_create(self.current_user['id'], public=False, name=playlist_name)
        webbrowser.open(self.playlist['uri'])

        if self.interactive:
            self.fill_playlist_interactive()
        else:
            self.fill_playlist_automatic()
        
        print('>> end of playlist creation')
        return

    # --- UPDATED ---
    def fill_playlist_automatic(self):
        print("Generating NLP-based playlist...")
        
        # Asks our new NLP function to generate the playlist contents
        self.generated_tracks = self.generate_nlp_playlist()
        first_song = None
        
        if not self.generated_tracks:
            print("No tracks to add. Exiting.")
            return

        for song in self.generated_tracks:
            track_id = song['id']
            self.sp.user_playlist_add_tracks(self.current_user['id'], self.playlist['id'], [track_id])
            print(f"Added: {song['artists'][0]['name']} - {song['name']}")
            
            if not first_song:
                first_song = song
        
        if first_song:
            webbrowser.open(self.playlist['uri'])
            self.play_song_in_spotify(first_song)

    def play_song_in_spotify(self, song, start_position=0):
        # ... (This function is unchanged) ...
        try:
            devices = self.sp.devices().get('devices', [])
            if not devices:
                print("No active Spotify device found. Trying to open Spotify...")
                try:
                    subprocess.Popen(['open_spotify.bat', song['uri']])
                except FileNotFoundError:
                    print("Could not find 'open_spotify.bat'. Opening playlist in browser instead.")
                    webbrowser.open(self.playlist['uri'])
                except Exception as e:
                    print(f"Error opening Spotify: {e}. Opening in browser.")
                    webbrowser.open(self.playlist['uri'])
            else:
                print(f"Playing on device: {devices[0]['name']}")
                self.sp.transfer_playback(devices[0]['id'])
                self.sp.start_playback(uris=[song['uri']], position_ms=start_position)
        except Exception as e:
            print(f"Error playing song: {e}")
            webbrowser.open(self.playlist['uri'])

    # --- UPDATED ---
    def fill_playlist_interactive(self):
        track_id = 0
        build_playlist = True
        first_song = None

        # Asks our new NLP function to generate the playlist contents
        self.generated_tracks = self.generate_nlp_playlist()
        
        if not self.generated_tracks:
            print("No tracks to add. Exiting.")
            return

        for song in self.generated_tracks: # Loop through the sorted, full track objects
            if not build_playlist:
                break
            
            track_id += 1
            artist_name = song['artists'][0]['name']
            song_name = song['name']
            track_name = f'{artist_name} - {song_name}'
            
            print(separator)
            
            if track_name in self.playlist_tracks:
                print(f'Ignore {song_name} | already in playlist')
                continue 
            if artist_name in self.artists_blacklist:
                print(f'Ignore {artist_name} | blacklisted')
                continue
            if song_name in self.songs_blacklist:
                print(f'Ignore {song_name} | blacklisted')
                continue
            
            track = self.sp.track(song['id']) # Get fresh track details if needed
                                
            if not first_song:
                first_song = song

            start_position = track['duration_ms'] / 2
            self.play_song_in_spotify(song, start_position)
            
            print(f'{track_name}  ({track_id}/{len(self.generated_tracks)})')
            print('[1] Add to Playlist')
            print('[2] Not this song')
            print('[3] Not this artist')
            print('[q] Quit playlist generation')
            
            while True:
                match input('Your choice: '):
                    case '1':
                        self.sp.user_playlist_add_tracks(self.current_user['id'], self.playlist['id'], [track['id']])
                        self.playlist_tracks.add(track_name)
                        self.songs_in_playlist.add(song_name)
                        print(f'>> added to playlist ({len(self.playlist_tracks)} tracks)')
                        break
                    case '2':
                        self.songs_blacklist.add(song_name)
                        print('>> song blacklisted')
                        break
                    case '3':
                        self.artists_blacklist.add(artist_name)
                        print('>> artist blacklisted')
                        break
                    case 'q':
                        build_playlist = False
                        break
                    case _:
                        print('Invalid choice. Please enter 1, 2, 3 or q to quit')
        
        print(separator)
        print('>> end of playlist creation')

        if first_song:
            print("Playing the first song added to the playlist.")
            webbrowser.open(self.playlist['uri'])
            self.play_song_in_spotify(first_song)


if __name__ == '__main__':
    
    if not args.p:
        prompt = input("Enter a prompt (e.g., 'sad rainy day songs', '80s rock hits'): ")
        if not prompt:
            print("A prompt is required.")
            exit()
        args.p = prompt
        if 'n' not in args: 
            args.n = args.p
    
    prompt = args.p
    length = args.l
    name = args.n
    interactive = args.i

    print(separator)
    print(' SPOTIFY DEEP LEARNING PLAYLIST GENERATOR '.center(100, '-'))
    print(separator)
    
    if not prompt:
        print('Error: You must provide a prompt using -p "your prompt"')
        parser.print_help()
    else:
        print('Creating playlist. Please wait...')
        playlist = SpotifyPlaylist(prompt, length, name, interactive)
        playlist.main()
        print(playlist)