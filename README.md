# Ultra-Accurate Face Uploader

A Python script for automated face recognition and Google Drive uploads.  
It processes new images, detects and identifies faces against a set of known faces, and uploads matched images to their respective Google Drive folders.

---

## Features

- Detects and recognizes faces using [DeepFace](https://github.com/serengil/deepface).
- Supports caching of embeddings for faster runs.
- Automatically uploads recognized faces to mapped Google Drive folders.
- Generates logs for tracking successes, warnings, and errors.

---

## Requirements

- Python 3.8 or higher
- Google Drive account with folder structure set up
- Google Cloud OAuth credentials file (`client_secrets.json`)

### Install Dependencies

```bash
pip install deepface pydrive pillow numpy tqdm
```

--------------------------------------------------------------------------------

Folder Structure

Set up your project directory as follows:

driveuploader/
│
├── script/
│   ├── uploader.py              # The script
│   └── client_secrets.json      # Google OAuth credentials
│
├── known_faces/                 # Reference photos of known people
│   ├── john_robinson.jpg
│   ├── jane_doe.png
│   └── ...
│
└── new_images/                  # New photos to be processed

--------------------------------------------------------------------------------

Preparing Known Faces

Add 10 or more clear images per person in the known_faces/ folder.

File names are normalized to IDs. For example:

John Robinson.jpg → john_robinson

A cache of embeddings will be created for faster future runs.

--------------------------------------------------------------------------------

Configuring Google Drive:

In the script, update the mapping between people and their Google Drive folder IDs:

person_to_folder_id = {
    "john_robinson": "GOOGLE_DRIVE_FOLDER_ID_FOR_JOHN_ROBINSON",
    "jane_doe": "GOOGLE_DRIVE_FOLDER_ID_FOR_JANE_DOE"
}


To find the folder ID in Google Drive, open the folder and copy the value from the URL:

https://drive.google.com/drive/folders/<FOLDER_ID>

--------------------------------------------------------------------------------

Running the Script

From the script/ directory, run:

python uploader.py

--------------------------------------------------------------------------------

The script will:

Authenticate with Google Drive (browser window opens on first run).

Process images from new_images/.

Detect and identify faces using known embeddings.

Upload matched images to the corresponding Google Drive folders.

--------------------------------------------------------------------------------

Logs

A log file named drive_uploader.log will be generated in the working directory.
It contains detailed information about authentication, embedding, detection, matches, and uploads.

--------------------------------------------------------------------------------

Example Workflow

Add john_robinson.jpg to known_faces/

Add party_photo.png to new_images/

Run the script

If John is detected in the photo, party_photo.png is uploaded to his mapped Google Drive folder.

--------------------------------------------------------------------------------

Notes

Images with no confident matches are not uploaded.

If no faces are detected, the file is skipped.

Cached embeddings allow repeated runs without recomputation.

Logs are persistent across runs for traceability.

_________________________________________________________________________________
