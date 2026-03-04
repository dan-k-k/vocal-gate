# ml_pipeline/prepare_data.py
import os
import csv
import shutil
import random
import librosa
import soundfile as sf

# Directories
ESC50_DIR = "./ESC-50-master"
LIBRISPEECH_DIR = "./LibriSpeech/dev-clean"
FSD50K_DIR = "./FSD50K"
DATA_DIR = "./data"

# Mappings
ARTIFACT_MAP = {
    '21': 'sneezing', 
    '23': 'breathing', 
    '24': 'coughing'
}

FSD50K_ARTIFACT_TAGS = [
    "Breathing", "Burping_and_eructation", "Cough", "Gasp", 
    "Respiratory_sounds", "Sigh", "Sneeze", "Computer_keyboard", 
    "Typing", "Crumpling_and_crinkling", "Thump_and_thud", 
    "Knock", "Zipper_(clothing)"
]

FSD50K_EXCLUDED_VOCALS = [
    "Speech", "Child_speech_and_kid_speaking", "Female_speech_and_woman_speaking", 
    "Male_speech_and_man_speaking", "Conversation", "Human_voice", 
    "Singing", "Female_singing", "Male_singing", "Speech_synthesizer", 
    "Whispering", "Yell", "Screaming", "Shout", "Laughter", 
    "Crying_and_sobbing", "Chatter", "Crowd"
]

def create_directories():
    """Creates the hierarchical train/val/test folder structure."""
    if os.path.exists(DATA_DIR):
        print(f"🧹 Clearing existing {DATA_DIR} folder...")
        shutil.rmtree(DATA_DIR)
        
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(DATA_DIR, split, "vocals", "speech"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, "artifacts", "fsd50k_noise"), exist_ok=True)
        for name in ARTIFACT_MAP.values():
            os.makedirs(os.path.join(DATA_DIR, split, "artifacts", name), exist_ok=True)
            
    print("✅ Created target folder structure (Train / Val / Test).")

def gather_and_split_data():
    """Gathers and routes files into Train/Val/Test dicts without data leakage."""
    
    # Initialize our master routing dictionary
    splits = ['train', 'val', 'test']
    categories = list(ARTIFACT_MAP.values()) + ['speech', 'fsd50k_noise']
    data_dict = {split: {cat: [] for cat in categories} for split in splits}
    
    # --- 1. LibriSpeech (Split by Speaker Folders) ---
    print(f"🔍 Routing LibriSpeech (by Speaker ID)...")
    speaker_folders = [f for f in os.listdir(LIBRISPEECH_DIR) if os.path.isdir(os.path.join(LIBRISPEECH_DIR, f))]
    random.seed(42)
    random.shuffle(speaker_folders)
    
    n_speakers = len(speaker_folders)
    train_spks = set(speaker_folders[:int(n_speakers * 0.70)])
    val_spks = set(speaker_folders[int(n_speakers * 0.70):int(n_speakers * 0.85)])
    # The rest go to test
    
    for root, _, files in os.walk(LIBRISPEECH_DIR):
        # Identify which speaker this root folder belongs to
        speaker_id = os.path.basename(os.path.dirname(root)) if root != LIBRISPEECH_DIR else None
        if not speaker_id:
            continue
            
        # Route the files based on the speaker's assigned split
        split_target = 'train' if speaker_id in train_spks else ('val' if speaker_id in val_spks else 'test')
        
        for file in files:
            if file.endswith('.wav'):
                data_dict[split_target]['speech'].append(os.path.join(root, file))

    # --- 2. ESC-50 (Split by Official Metadata Folds) ---
    print(f"🔍 Routing ESC-50 (by Official Folds)...")
    esc50_csv = os.path.join(ESC50_DIR, "meta", "esc50.csv")
    esc50_audio = os.path.join(ESC50_DIR, "audio")
    
    with open(esc50_csv, 'r') as f:
        for row in csv.DictReader(f):
            target = row['target']
            if target in ARTIFACT_MAP:
                fold = int(row['fold'])
                # Folds 1, 2, 3 -> Train | Fold 4 -> Val | Fold 5 -> Test
                if fold in [1, 2, 3]:
                    split_target = 'train'
                elif fold == 4:
                    split_target = 'val'
                else:
                    split_target = 'test'
                    
                file_path = os.path.join(esc50_audio, row['filename'])
                data_dict[split_target][ARTIFACT_MAP[target]].append(file_path)

    # --- 3. FSD50K (Split by Dev/Eval sets) ---
    print("🔍 Routing FSD50K (by Dev/Eval Sets)...")
    target_tags = set(FSD50K_ARTIFACT_TAGS)
    excluded_tags = set(FSD50K_EXCLUDED_VOCALS)
    
    # Process Eval Set (100% goes to Test)
    eval_csv = os.path.join(FSD50K_DIR, "FSD50K.ground_truth", "eval.csv")
    eval_audio = os.path.join(FSD50K_DIR, "FSD50K.eval_audio_16k")
    if os.path.exists(eval_csv):
        with open(eval_csv, 'r') as f:
            for row in csv.DictReader(f):
                file_labels = set(row['labels'].split(','))
                if bool(file_labels & target_tags) and not bool(file_labels & excluded_tags):
                    file_path = os.path.join(eval_audio, f"{row['fname']}.wav")
                    if os.path.exists(file_path):
                        data_dict['test']['fsd50k_noise'].append(file_path)

    # Process Dev Set (Use official train/val split from CSV to prevent leakage)
    dev_csv = os.path.join(FSD50K_DIR, "FSD50K.ground_truth", "dev.csv")
    dev_audio = os.path.join(FSD50K_DIR, "FSD50K.dev_audio_16k")
    
    if os.path.exists(dev_csv):
        with open(dev_csv, 'r') as f:
            for row in csv.DictReader(f):
                file_labels = set(row['labels'].split(','))
                # Check if it has our target tags and NO excluded vocal tags
                if bool(file_labels & target_tags) and not bool(file_labels & excluded_tags):
                    file_path = os.path.join(dev_audio, f"{row['fname']}.wav")
                    if os.path.exists(file_path):
                        # Route using the official FSD50K split column!
                        official_split = row['split'] 
                        if official_split in ['train', 'val']:
                            data_dict[official_split]['fsd50k_noise'].append(file_path)
                        
    return data_dict

def copy_or_resample(src, dest_folder, target_sr=16000):
    """Copies a file if it matches target_sr and mono, otherwise resamples it."""
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, os.path.basename(src))
    
    info = sf.info(src)
    if info.samplerate == target_sr and info.channels == 1:
        shutil.copy(src, dest_path)
    else:
        y, sr = librosa.load(src, sr=target_sr, mono=True)
        sf.write(dest_path, y, sr)

def process_and_export(data_dict):
    """Iterates through our routed dictionary and exports everything."""
    print("⏳ Processing, conforming to 16kHz Mono, and copying files...")
    
    for split, categories in data_dict.items():
        print(f"\n📁 Processing {split.upper()} set...")
        for category, file_list in categories.items():
            if not file_list:
                continue
                
            parent_folder = "vocals" if category == "speech" else "artifacts"
            dest_folder = os.path.join(DATA_DIR, split, parent_folder, category)
            
            for file_path in file_list:
                copy_or_resample(file_path, dest_folder)
                
            print(f"   -> {category.capitalize()}: {len(file_list)} files")

if __name__ == "__main__":
    print("🚀 Starting Leak-Free Data Preparation Pipeline...")
    create_directories()
    routed_data = gather_and_split_data()
    process_and_export(routed_data)
    print("\n🎉 Pipeline finished! Your data is perfectly isolated and ready for training.")

