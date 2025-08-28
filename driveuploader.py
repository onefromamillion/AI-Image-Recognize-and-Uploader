import os
import re
import pickle
import hashlib
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import logging
from PIL import Image

# Setup logging
logging.basicConfig(
    filename='drive_uploader.log',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
logger.info("Starting ultra-accurate face uploader script...")

# Authenticate Google Drive
try:
    logger.info("Authenticating with Google Drive...")
    gauth = GoogleAuth()
    gauth.DEFAULT_SETTINGS['client_config_file'] = r"C:\\Users\\Artyom\\Desktop\\driveuploader\\script\\client_secrets.json"
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    logger.info("Drive authentication successful.")
except Exception as e:
    logger.error(f"Google Drive authentication failed: {e}")
    raise e

# Normalize name
def normalize_name(name):
    name = name.lower().strip()
    name = re.sub(r'\s*\(\d+\)$', '', name)  # Remove (1), (2) suffixes
    name = re.sub(r'\d+$', '', name)         # Remove trailing numbers
    name = name.replace(" ", "_")
    return name

# Compute folder hash
def compute_folder_hash(folder_path):
    h = hashlib.md5()
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            with open(os.path.join(folder_path, fn), 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
    return h.hexdigest()

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Identify face by voting
def identify_face(embedding, known_faces, min_votes=1, threshold=0.6, margin=0.04):
    scores = {}
    for name, embeddings in known_faces.items():
        sims = [cosine_similarity(embedding, e) for e in embeddings]
        votes = [s for s in sims if s >= threshold]
        if votes:
            scores[name] = {"votes": len(votes), "avg": np.mean(votes)}
    if not scores:
        return None
    best = max(scores.items(), key=lambda x: (x[1]["votes"], x[1]["avg"]))
    name, stats = best
    if stats["votes"] < min_votes or stats["avg"] < threshold:
        return None
    sorted_scores = sorted(scores.values(), key=lambda s: (s["votes"], s["avg"]), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0]["avg"] - sorted_scores[1]["avg"] < margin:
        return None
    return name

# Load known faces
known_faces_folder = "known_faces"
cache_file = "known_faces_cache.pkl"
hash_file = "known_faces_cache.hash"
known_faces = {}

folder_hash = compute_folder_hash(known_faces_folder)
use_cache = (
    os.path.exists(cache_file) and
    os.path.exists(hash_file) and
    open(hash_file).read().strip() == folder_hash
)

if use_cache:
    logger.info("Loading known face embeddings from cache...")
    with open(cache_file, 'rb') as f:
        known_faces = pickle.load(f)
else:
    logger.info("Computing known face embeddings from scratch...")
    for fn in tqdm(os.listdir(known_faces_folder), desc="Embedding known faces"):
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = normalize_name(os.path.splitext(fn)[0])
            path = os.path.join(known_faces_folder, fn)
            try:
                emb = DeepFace.represent(img_path=path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
                known_faces.setdefault(name, []).append(np.array(emb))
                logger.info(f"Embedded {name}")
            except Exception as e:
                logger.warning(f"Failed to embed {fn}: {e}")
    with open(cache_file, 'wb') as f:
        pickle.dump(known_faces, f)
    with open(hash_file, 'w') as f:
        f.write(folder_hash)
    logger.info("Embeddings cached.")

logger.info(f"{sum(len(v) for v in known_faces.values())} embeddings loaded for {len(known_faces)} people.")

# Your folder id map (replace with your actual folder IDs)
person_to_folder_id = {
  "alen_simonyan": "GOOGLE_DRIVE_FOLDER_ID_FOR_ALEN_SIMONYAN",
  "hakob_arshakyan": "1WwVfK1GXIYnDWuJqJewJiXec9TPszy8I",
  "ruben_rubinyan": "1rlphf9NCf8mftkmjKAGcKj7SWqDFH6xm",
  "tigran_abrahamyan": "1cC2A2--OvFzW6Pctc4My6NYIvnkPq2MN",
  "julieta_azaryan": "16gZHWeXkm9OT6ed93OlwmBo1WWcoO2c5",
  "vahagn_aleksanyan": "1YsQ-xIlI7UXhQaqHZ6S1qM1-4CgYVpKF",
  "hovik_aghazaryan": "15isn3OIvepgHLMDrruIPSBra4hnhgGs7",
  "eduard_aghajanyan": "1EyeXbygGB6biM-Fib2ZVLyMoF2k6SpH4",
  "hakob_aslanyan": "1KUY3xuCunP3Z26eqayOiPJtkcYmjAd0l",
  "aleksandr_avetisyan": "1qIB-PB_GCY0bwLH6urNN-fS2AaPNJi3d",
  "tadevos_avetisyan": "1Id0b8Jv67VcLCNOEbykFFu8LCvmW5MSH",
  "narek_babayan": "1q7H-alLWtEmwG0j6Qx1tQxjdIWER-z3g",
  "sergey_bagratyan": "1rP1hCbCwBH1oMMWlFD5TdzUkyYH0mPfj",
  "lusine_badalyan": "1uM91B-CNfFaRJlxKTcXjOl-wHClqIUv5",
  "zaruhi_batoyan": "10aAI5O-FsLLDuUnPyVOxqnbPcPAoBhu7",
  "rustam_bakoyan": "1b5BCOt_6wozhDPb7H4Vt_EaS5Yz-DoZ3",
  "sisak_gabrielyan": "1S3jkNgvveAekhEqn2nYH7l71EYvCqqeL",
  "lilit_galstyan": "1cr023h3Xrmqcb3RIThhWdj5scZ51sFvV",
  "meri_galstyan": "1Dgk6ztkqAEhFSAWY8OWL0hO4CNLN3orn",
  "irina_gasparyan": "1ZtTpSP6hawpKR8vb3vrT8mqv7qD2jfiy",
  "tatevik_gasparyan": "1trzIzuexrkKNsbnFgp0j8u4i5qtK7tdZ",
  "argishti_gevorgyan": "1kSytj2Q3tZnbzVsDDNCVxvca9wwk0IYm",
  "armen_gevorgyan": "1KZCD4by9ZPQbo6MlDIgEZ1ixMM6Yvb4E",
  "anna_grigoryan": "1LNeCFuE8ulwQ0sfCMBkdvak_DkVLtCEG",
  "hripsime_grigoryan": "1r0O7-yXqPFUUoWGktFrK-VaA_8_vXyyw",
  "martun_grigoryan": "1cSkOGdMX7QmF_MNwtAjxZyLIcQ89GpuK",
  "narek_grigoryan": "1HoWmo3BhhJpnH7Nbn2PUr4bJ1IsFs6Bb",
  "garnik_danielyan": "137YMsTO8zYkYt47CftuS4TgSxLIFU-_A",
  "davit_danielyan": "1fFjo3y5S1tdOnEIesFr6fgPJyFBH_UWj",
  "arpine_davoyan": "11hqNxaEMxjVGQ3ZIuxvQLj-WAOrfXatZ",
  "gayane_eghiazaryan": "1qT6ODtT6iP9GMcIUvrIXXeoZ9KhkxcXJ",
  "arman_eghoyan": "1gjZbXZA_nND3OGc4c8pX66-p1n5kHBGT",
  "ishkhan_zakaryan": "18RfqkQdEu29iBu9i1TATGSXeHSVR4zRr",
  "taguhi_tovmasyan": "1182lKmKRlFPRekA0XV-sR7FOyd919lkt",
  "arsen_torosyan": "1FvmLjx0Nf-8kifIfm52XmGthN4T6B-Xk",
  "shirak_torosyan": "1K07WPiqp8dVtBfTVkCqwPzVBJyH7m0P5",
  "mikayel_tumasyan": "1diHfRWJYZJCK31ysg-1Ue4RGyO89X923",
  "babken_tunyan": "1SHkP8raDq2U6kDM4V-IHmV27Sv4jyQH_",
  "agnesa_khamoyan": "1BZxBpu5Mzg1lwil2pl_yq_YWHpYPSdxB",
  "sargis_khandanyan": "1-vfi9wbty6C0dAYg2Ch9sLlmGubsxQ7t",
  "artur_khachatryan": "1zkaMH20jiXctXcRstnMuAJYfG695abZ7",
  "armen_khachatryan": "1KIoMTUAsFnapShDL2VW4VqX8B8yBH-oC",
  "armine_kheranyan": "1kIxi3CQPCgXeC2K3TWBxJhh7D2CrE6lx",
  "davit_karapetyan": "1HqzCglyYaks1gmFHvElUUNpUVI5sF2Ke",
  "maria_karapetyan": "1BACGLbaiTglu5Gyv6uA-7jpjO-v4xJHZ",
  "lilit_kirakosyan": "1zMm9tQ-mQqiC2sF0b-iPmJuqKhV-e9rP",
  "armenui_kyureghyan": "1CAmsG8wTuUwr6KSGylo5KRRpStlcBVbL",
  "hayk_konjoryan": "1s1w3cbqtCd3691b_KTlgbShcFM4GB0jA",
  "aspram_krpeyan": "1-dpRR2Wb68HqIwAJFM8_UGuYS--2Ts7h",
  "edgar_hakobyan": "1Sg-RB-8J8zjeBLqrYhrMGID9IxdlkJjS",
  "hasmik_hakobyan": "1e035DBAzj4KLGy-Dhaj31yDntGe1ae_O",
  "hrachya_hakobyan": "1-ICqZtnZu7-KAuQvvtD_Grl1ysGKtVLS",
  "vagharshak_hakobyan": "1B1gg00xLKz3Ry2LI1CDbbRxavlrGsPI8",
  "karen_hambardzumyan": "1ELolxfl2_j4SVmfdGyUn0kippP5ho10L",
  "knyaz_hasanyan": "1QW881d9Vw7zJPBCX_dr-LH6uQNt_c08w",
  "artur_hovhannisyan": "1ODgDiEKJ6XgZyYNyPm684vFLqbTowYqH",
  "hripsime_hunanyan": "1wHysCIXORug-t3I0lvznBevhw6bU3LEI",
  "alkhas_ghazaryan": "1LNzeODDtrNCz6uBT41heKRLIICxWAYuA",
  "arman_ghazaryan": "18yypXxE0jxnJsbSqQolf5Fcu8fD-I5zd",
  "armenui_ghazaryan": "1-17M_l-sqESq4GljyqogWEeIPFbpuGEv",
  "taguhi_ghazaryan": "151WQA4pyp4Upc3YaRf5NAxl9fhJ_D-Sa",
  "marina_ghazaryan": "13Gx60NideepjpNrxJwXF3FyMJKBWhbEm",
  "sona_ghazaryan": "1CzdDw626m9YFVM6KWaR4Z2nlxbzaDqy8",
  "vahe_ghalumyan": "1WHS5ItJAVYoQOJ8AgHjVALTW7FtQHSmc",
  "narek_gahramanyan": "1PCUb_nzquGzoL0q0e8rfPXOf_y-LJJJj",
  "hayk_mamijanyan": "1azwGm1q3jf5B8U9ZqDnsrKH1y2xPoamx",
  "arusyak_manavazyan": "1zWt7CkDralxvzwFxzw6I3y0NYgd4ce56",
  "aregnaz_manukyan": "1T18zA9Zs5qzcHm4Cl7_BTpLp1CDWOq90",
  "gegham_manukyan": "1qqBR2ELxCkIU3gS_gLEl-8Jpn0OcBjN9",
  "taron_margaryan": "18rZrfU7avSuagRpbg8VxSxsULPWPb2MK",
  "gurgen_melkonyan": "1fw4V3HoCwcNLVp160ZsfBCEJlYxs4Ujf",
  "mher_melkonyan": "1aWrAL3KPvC0pbI1ievLXY9ingeoa80kN",
  "artyom_mehrabyan": "10M4vEW8s3Y-JC2aSOfLeqDRoabQ7d0lt",
  "artsvik_minasian": "1DdXy6MrKc-Aj-J2nIiQ-qBdbumZn_5Nt",
  "lilit_minasyan": "1brk9-WxnGiRrSY9IUVJ0yAmeFhpD5f-T",
  "zemfira_mirzoyeva": "1YseJ5TVK5SbX9_6S9U62bSgm6AlOWU6o",
  "anna_mkrtchyan": "16kOhZ2DUyuNzqkkD8bnisrftfTB0FY0M",
  "aren_mkrtchyan": "1AWkuzsQPwlSg-yfKPxU9ykVcs1FMGl6v",
  "gegham_nazaryan": "1LL8CctX3w60DyQ7CNg0MfXcgp9L1bC6r",
  "lena_nazaryan": "1v35_hudUaK3Jq8QQseMSDpdJYQox2qEd",
  "emma_palyan": "1KFagTSNi51YoageyQ_zyF0oyu1dKXNAz",
  "tigran_parsilian": "15Kas4EvncEDyXNsz39HaImSdH3w5NKYe",
  "mariam_poghosyan": "12b082an1Rs9zxN-K8HYk9S62PqMIkHOU",
  "kristine_poghosyan": "1pQ9GwWxO1azx0_7SdO_OHDFH77j_mfRl",
  "arusyak_julhakyan": "10uT8snkHr7OD6XPCKUnn_2pibZc3Gzrn",
  "armen_rustamyan": "1Bw4IPyJzA8tF-i-PLHpXTleBVF-0joGQ",
  "ishkhan_saghatelian": "19Mkae2oSbEhP7CVO8VQ7ONjO3lfR9AWL",
  "aleksey_sandikov": "1arNJdEJ6JRZ8hMwAHMzebwXmh-JToYtk",
  "artur_sargsyan": "1yupsRmprrYPKHfYenmNl4vhA0-6pSUNR",
  "hayk_sargsyan": "1RewsScPQMzAl5pNvQokqBDs3DTQHghuD",
  "trdat_sargsyan": "13MWsoObQAO_Myph09nU0Ov9u0hFXo19J",
  "karen_sarukhanyan": "1ICVLAoI85eYWSwWzAqtuKS4YvEOLbyhN",
  "ashot_simonyan": "1b47zvRXcW6dE-aXpyckzhXZgn-Qyj-Og",
  "khachatur_sukiasyan": "1Eiu1TcjJP3mDgIckPjKUCIZoL01hIOFZ",
  "lilit_stepanyan": "1PFNmc4JsbaWJTuPyeywjLgIR-xd4mZOj",
  "aghvan_vardanyan": "17oidCQeYL4AxBdAh1q6uFIJSwdlm8o3p",
  "arpine_vardanyan": "1E8HsciiyPX5sS-MD9C3hjg0hfk6zrz7D",
  "elinar_vardanyan": "1pq7Zxv_PUwAyUqmi-LZ2qK1O3CN0JLY4",
  "tsovinar_vardanyan": "1HK00lf3EP0lnNStVjY5y7S1NZ2CqkxpY",
  "vladimir_vardanyan": "1y5jRgPQdLG_-LmGRzBPt6WBpsWFY2IwH",
  "kristine_vardanyan": "17oidCQeYL4AxBdAh1q6uFIJSwdlm8o3p",
  "heriknaz_tigranyan": "1BDS30cdoaMlj42AIlCP7ArE6swKH88sA",
  "hayk_tsirunyan": "18btn4Nw7smrgnH--GuC3bSFDguWZY5Jn",
  "anush_kloyan": "1uJ_t3pfIUqm4O5xrKcrNB63JhkccfpXo",
  "andranik_kocharyan": "14TVfyZdWgfxfp90tEzQ6JXjLDdD9kSE_",
  "levon_kocharyan": "1uPrLShZxYIQ2HxWY-GTuCgM5QD9SVuMn",
  "seyran_ohanyan": "1clKiypOtgIVw7kHDaQKAiPi-G0KxrAX4"
}

# Process new images
new_images_dir = "new_images"
new_images = [f for f in os.listdir(new_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not new_images:
    logger.info("No new images found.")
    exit()

logger.info(f"Processing {len(new_images)} new images...")

for fn in tqdm(new_images, desc="Processing images"):
    path = os.path.join(new_images_dir, fn)
    try:
        faces = DeepFace.extract_faces(img_path=path, detector_backend='opencv', enforce_detection=False)
    except Exception as e:
        logger.error(f"Face detection failed for {fn}: {e}")
        continue

    if not faces:
        logger.warning(f"No faces found in {fn}")
        continue

    matched = set()
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {fn}: {e}")
        continue

    for face in faces:
        try:
            fa = face["facial_area"]  # {'x':int, 'y':int, 'w':int, 'h':int}
            face_crop = img.crop((fa['x'], fa['y'], fa['x'] + fa['w'], fa['y'] + fa['h']))
            temp_face_path = f"temp_{fn}"
            face_crop.save(temp_face_path)

            emb = DeepFace.represent(img_path=temp_face_path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
            os.remove(temp_face_path)

            match = identify_face(np.array(emb), known_faces)
            if match:
                matched.add(match)
        except Exception as e:
            logger.warning(f"Embedding failed for face in {fn}: {e}")

    if matched:
        for person in matched:
            folder_id = person_to_folder_id.get(person)
            if not folder_id:
                logger.warning(f"No folder found for: {person}")
                continue
            try:
                file = drive.CreateFile({'title': fn, 'parents': [{'id': folder_id}]})
                file.SetContentFile(path)
                file.Upload()
                logger.info(f"Uploaded {fn} to {person}")
            except Exception as e:
                logger.error(f"Upload failed for {fn} to {person}: {e}")
    else:
        logger.info(f"No confident match for '{fn}'")

logger.info("All done.")
