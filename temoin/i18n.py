"""French/English UI strings for the Streamlit interface."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "app_title": {
        "fr": "Témoin — témoin visuel neutre",
        "en": "Témoin — neutral visual witness",
    },
    "tagline": {
        "fr": "Prouvez que l'objet livré est bien celui qui a été vendu.",
        "en": "Prove the delivered item is the one that was sold.",
    },
    "disclaimer": {
        "fr": "Témoin assiste une décision humaine, il ne la remplace pas.",
        "en": "Témoin assists a human decision, it never replaces it.",
    },
    "tab_sale": {"fr": "Vente", "en": "Sale"},
    "tab_delivery": {"fr": "Livraison", "en": "Delivery"},
    "tab_verify": {"fr": "Vérifier", "en": "Verify"},
    "sale_new": {"fr": "Nouvelle vente", "en": "New sale"},
    "sale_start_hint": {
        "fr": "Créez une vente pour recevoir un code défi.",
        "en": "Create a sale to receive a challenge code.",
    },
    "sale_code_title": {
        "fr": "Code vendeur — écrivez-le sur un papier à côté de l'objet :",
        "en": "Seller code — write it on a paper next to the item:",
    },
    "sale_upload": {
        "fr": "Photo de l'objet avec le code",
        "en": "Photo of the item with the code",
    },
    "sale_analyze": {"fr": "Analyser la photo", "en": "Analyze the photo"},
    "sale_analyzing": {"fr": "Analyse en cours…", "en": "Analyzing…"},
    "report_object": {"fr": "Objet", "en": "Item"},
    "report_defects": {"fr": "Défauts relevés", "en": "Recorded flaws"},
    "report_none": {"fr": "Aucun défaut visible.", "en": "No visible flaws."},
    "report_checkpoints": {
        "fr": "Points de contrôle — à rephotographier à la livraison",
        "en": "Checkpoints — to re-photograph at delivery",
    },
    "photo_caption": {"fr": "Défauts encadrés", "en": "Boxed flaws"},
    "fresh_ok": {
        "fr": "Code vérifié (lu : {code})",
        "en": "Code verified (read: {code})",
    },
    "fresh_ko": {
        "fr": "Code non vérifié — reprenez la photo avec le code lisible",
        "en": "Code not verified — retake the photo with a readable code",
    },
    "sale_share_title": {"fr": "Envoyer à l'acheteur", "en": "Send to the buyer"},
    "sale_share_help": {
        "fr": "L'acheteur saisit cet identifiant dans l'onglet Livraison.",
        "en": "The buyer enters this id in the Delivery tab.",
    },
    "sale_share_text": {
        "fr": (
            "Témoin — Dossier {tx} : ouvrez Témoin, onglet Livraison, saisissez "
            "cet identifiant pour recevoir votre code et photographier l'objet."
        ),
        "en": (
            "Témoin — Case {tx}: open Témoin, Delivery tab, enter this id "
            "to receive your code and photograph the item."
        ),
    },
    "sale_whatsapp": {"fr": "Partager sur WhatsApp", "en": "Share on WhatsApp"},
    "sale_simulated": {
        "fr": "Lien de partage simulé (pas d'intégration WhatsApp Business).",
        "en": "Simulated share link (no WhatsApp Business integration).",
    },
    "severity_minor": {"fr": "léger", "en": "minor"},
    "severity_moderate": {"fr": "net", "en": "moderate"},
    "severity_major": {"fr": "grave", "en": "major"},
    "col_severity": {"fr": "Gravité", "en": "Severity"},
    "col_sale": {"fr": "Défaut vente", "en": "Sale flaw"},
    "col_delivery": {"fr": "Défaut livraison", "en": "Delivery flaw"},
    "col_status": {"fr": "Statut", "en": "Status"},
    "st_matched": {"fr": "Correspondant", "en": "Matched"},
    "st_missing": {"fr": "Absent", "en": "Missing"},
    "st_new": {"fr": "Nouveau", "en": "New"},
    "st_changed": {"fr": "Modifié", "en": "Changed"},
    "del_tx_label": {"fr": "Identifiant du dossier", "en": "Case identifier"},
    "del_hint": {
        "fr": "Saisissez l'identifiant reçu du vendeur, puis chargez le dossier.",
        "en": "Enter the id received from the seller, then load the case.",
    },
    "del_load": {"fr": "Charger le dossier", "en": "Load the case"},
    "del_not_found": {
        "fr": "Dossier introuvable. Vérifiez l'identifiant.",
        "en": "Case not found. Check the identifier.",
    },
    "del_no_sale": {
        "fr": "La vente n'a pas encore de constat.",
        "en": "The sale has no constat yet.",
    },
    "del_recap": {"fr": "Rappel de la vente", "en": "Sale recap"},
    "del_gen_code": {"fr": "Recevoir mon code livraison", "en": "Get my delivery code"},
    "del_code_title": {
        "fr": "Code livraison — écrivez-le sur un papier à côté de l'objet :",
        "en": "Delivery code — write it on a paper next to the item:",
    },
    "del_upload": {
        "fr": "Photo de l'objet reçu avec le code",
        "en": "Photo of the received item with the code",
    },
    "del_investigate": {"fr": "Lancer l'enquête", "en": "Start the investigation"},
    "del_investigating": {"fr": "Enquête en cours…", "en": "Investigating…"},
    "del_done": {"fr": "Enquête terminée", "en": "Investigation finished"},
    "del_verdict": {"fr": "Verdict", "en": "Verdict"},
    "del_confidence": {"fr": "Confiance", "en": "Confidence"},
    "del_justification": {"fr": "Justification", "en": "Justification"},
    "del_matches": {"fr": "Correspondances des défauts", "en": "Flaw matches"},
    "del_contradictions": {"fr": "Contradictions", "en": "Contradictions"},
    "del_missing_proof": {
        "fr": "Preuves manquantes — photos à reprendre",
        "en": "Missing proof — photos to retake",
    },
    "alerts": {"fr": "Alertes", "en": "Alerts"},
    "del_trail": {"fr": "Trace d'enquête", "en": "Investigation trail"},
    "del_certificate": {"fr": "Certificat scellé", "en": "Sealed certificate"},
    "del_download": {
        "fr": "Télécharger le certificat (JSON)",
        "en": "Download the certificate (JSON)",
    },
    "del_qr": {"fr": "QR code de partage", "en": "Share QR code"},
    "sam_title": {
        "fr": "Segmentation précise (SAM 3.1)",
        "en": "Precise segmentation (SAM 3.1)",
    },
    "sam_concept": {
        "fr": "Objet à détourer, en anglais (ex. headphones)",
        "en": "Object to outline (e.g. headphones)",
    },
    "sam_button": {"fr": "Segmenter la photo", "en": "Segment the photo"},
    "sam_running": {"fr": "Segmentation en cours…", "en": "Segmenting…"},
    "sam_need_concept": {
        "fr": "Indiquez d'abord l'objet à détourer.",
        "en": "Enter the object to outline first.",
    },
    "sam_none": {
        "fr": "Aucun « {concept} » trouvé sur la photo.",
        "en": "No “{concept}” found in the photo.",
    },
    "sam_caption": {
        "fr": "Masques SAM — contours exacts",
        "en": "SAM masks — exact outlines",
    },
    "ver_upload": {
        "fr": "Déposer un certificat (JSON)",
        "en": "Drop a certificate (JSON)",
    },
    "ver_check": {"fr": "Vérifier", "en": "Verify"},
    "ver_no_file": {
        "fr": "Déposez d'abord un fichier.",
        "en": "Drop a file first.",
    },
    "ver_invalid": {
        "fr": "Ce fichier n'est pas un certificat Témoin valide.",
        "en": "Not a valid Témoin certificate.",
    },
    "ver_intact": {
        "fr": "INTACT — ce certificat n'a pas été modifié.",
        "en": "INTACT — this certificate was not modified.",
    },
    "ver_tampered": {
        "fr": "MODIFIÉ — ce certificat ne correspond plus à son sceau.",
        "en": "TAMPERED — this certificate no longer matches its seal.",
    },
    "ver_sealed_on": {"fr": "Scellé le", "en": "Sealed on"},
    "ver_photos": {"fr": "Photos", "en": "Photos"},
    "ver_steps": {"fr": "étapes journalisées", "en": "logged steps"},
    "ver_link_first": {"fr": "premier maillon", "en": "first link"},
    "ver_link_chained": {"fr": "chaîné au précédent", "en": "chained to previous"},
    "ver_link": {"fr": "Maillon", "en": "Link"},
    "ver_chain_title": {
        "fr": "Ou vérifier une chaîne par identifiant",
        "en": "Or verify a chain by identifier",
    },
    "ver_tx_label": {"fr": "Identifiant du dossier à vérifier", "en": "Case id to verify"},
    "ver_chain_check": {"fr": "Vérifier la chaîne", "en": "Verify the chain"},
    "ver_chain_ok": {
        "fr": "Chaîne intacte : {n} certificat(s) vérifié(s).",
        "en": "Chain intact: {n} certificate(s) verified.",
    },
    "ver_chain_ko": {
        "fr": "Chaîne ROMPUE : au moins un certificat est invalide.",
        "en": "Chain BROKEN: at least one certificate is invalid.",
    },
    "ver_no_certs": {
        "fr": "Aucun certificat scellé dans ce dossier.",
        "en": "No sealed certificate in this case.",
    },
    "err_photo_first": {"fr": "Déposez d'abord une photo.", "en": "Upload a photo first."},
    "tab_video": {"fr": "Vidéo", "en": "Video"},
    "vid_hint": {
        "fr": "Filmez l'objet en tournant lentement autour, code visible, puis déposez la vidéo.",
        "en": "Film the item with a slow walkaround, code visible, then drop the video.",
    },
    "vid_code": {"fr": "Code défi montré dans la vidéo", "en": "Challenge code shown in the video"},
    "vid_upload": {"fr": "Vidéo du tour de l'objet", "en": "Walkaround video of the item"},
    "vid_analyze": {"fr": "Analyser la vidéo", "en": "Analyze the video"},
    "vid_running": {"fr": "Analyse des vues en cours…", "en": "Analyzing views…"},
    "vid_no_file": {"fr": "Déposez d'abord une vidéo.", "en": "Upload a video first."},
    "vid_need_code": {
        "fr": "Indiquez d'abord le code défi de la vidéo.",
        "en": "Enter the video challenge code first.",
    },
    "vid_views": {"fr": "Vues analysées", "en": "Analyzed views"},
    "vid_view": {"fr": "Vue", "en": "View"},
    "vid_fresh_ok": {
        "fr": "Code vu sur {seen}/{total} vues",
        "en": "Code seen on {seen}/{total} views",
    },
    "vid_fresh_ko": {
        "fr": "Code non vu — filmez le papier avec l'objet",
        "en": "Code not seen — film the paper with the item",
    },
    "vid_fused": {"fr": "Défauts confirmés multi-vues", "en": "Multi-view confirmed flaws"},
    "vid_confirmed": {"fr": "Confirmé sur", "en": "Confirmed on"},
    "sale_materials": {
        "fr": "Matières annoncées (séparées par des virgules)",
        "en": "Declared materials (comma-separated)",
    },
    "report_materials": {"fr": "Matières observées", "en": "Observed materials"},
    "report_declared": {"fr": "Matières annoncées", "en": "Declared materials"},
    "report_grade": {"fr": "État général", "en": "Overall condition"},
    "col_impact": {"fr": "Impact valeur", "en": "Value impact"},
    "impact_low": {"fr": "faible", "en": "low"},
    "impact_medium": {"fr": "modéré", "en": "medium"},
    "impact_high": {"fr": "fort", "en": "high"},
    "val_basis": {
        "fr": "Note calculée depuis {n} défaut(s) relevé(s).",
        "en": "Grade computed from {n} recorded flaw(s).",
    },
    "voice_title": {"fr": "Déclaration vocale", "en": "Voice declaration"},
    "voice_record": {"fr": "Enregistrer au micro", "en": "Record with microphone"},
    "voice_upload": {"fr": "Ou déposer une note WAV", "en": "Or drop a WAV note"},
    "voice_button": {"fr": "Transcrire", "en": "Transcribe"},
    "voice_running": {"fr": "Transcription en cours…", "en": "Transcribing…"},
    "voice_no_audio": {
        "fr": "Enregistrez ou déposez d'abord une note vocale.",
        "en": "Record or upload a voice note first.",
    },
    "voice_empty": {"fr": "Aucune parole détectée.", "en": "No speech detected."},
    "for_title": {"fr": "Vérifications photo", "en": "Photo checks"},
    "for_ok": {"fr": "Photo saine, bonne qualité.", "en": "Photo clean, good quality."},
    "tab_depth": {"fr": "Profondeur", "en": "Depth"},
    "depth_hint": {
        "fr": "Deux photos du même côté de l'objet, sous deux angles proches (un pas à droite).",
        "en": "Two photos of the same side of the item, from two close angles (one step right).",
    },
    "depth_left": {"fr": "Vue de gauche", "en": "Left view"},
    "depth_right": {"fr": "Vue de droite", "en": "Right view"},
    "depth_button": {"fr": "Mesurer le relief", "en": "Measure the relief"},
    "depth_running": {"fr": "Mesure du relief en cours…", "en": "Measuring relief…"},
    "depth_need": {
        "fr": "Déposez deux vues ou une vidéo pour mesurer le relief.",
        "en": "Drop two views or a video to measure the relief.",
    },
    "depth_video": {
        "fr": "Ou vidéo du tour de l'objet",
        "en": "Or walkaround video of the item",
    },
    "depth_frames": {
        "fr": "Vues utilisées : {a} et {b}.",
        "en": "Views used: {a} and {b}.",
    },
    "depth_coverage": {
        "fr": "Relief mesuré sur {pct} % de l'image.",
        "en": "Relief measured on {pct}% of the image.",
    },
    "depth_wiggle": {"fr": "Relief animé (parallaxe)", "en": "Animated relief (parallax)"},
    "depth_cloud": {
        "fr": "Nuage 3D — tournez l'objet avec la souris",
        "en": "3D cloud — rotate the item with the mouse",
    },
    "depth_unknown": {
        "fr": "Zones sans relief mesuré : laissées vides, jamais inventées.",
        "en": "Unmeasured zones: left empty, never invented.",
    },
    "cloud_unavailable": {
        "fr": "Nuage 3D indisponible : le module Plotly n'est pas installé ici.",
        "en": "3D cloud unavailable: the Plotly module is not installed here.",
    },
    "depth_method": {"fr": "Méthode", "en": "Method"},
    "depth_method_stereo": {"fr": "Stéréo mesurée (2 vues)", "en": "Measured stereo (2 views)"},
    "depth_method_mono": {"fr": "IA monoculaire (1 vue)", "en": "Monocular AI (1 view)"},
    "depth_mono_need": {
        "fr": "Déposez une photo ou une vidéo pour l'estimation IA.",
        "en": "Drop a photo or a video for the AI estimate.",
    },
    "depth_mono_note": {
        "fr": "Relief estimé par IA : indicatif, ne prouve rien à lui seul.",
        "en": "AI-estimated relief: indicative only, not proof by itself.",
    },
    "depth_mono_ok": {"fr": "Relief estimé par IA.", "en": "AI-estimated relief."},
    "board_title": {"fr": "Planche de preuves", "en": "Evidence board"},
    "board_sale": {"fr": "Vente", "en": "Sale"},
    "board_delivery": {"fr": "Livraison", "en": "Delivery"},
    "board_missing": {"fr": "non visible", "en": "not visible"},
    "board_full": {"fr": "vue entière", "en": "whole view"},
    "board_none": {
        "fr": "Aucune correspondance relevée.",
        "en": "No correspondences recorded.",
    },
    "ask_title": {"fr": "Interroger le dossier", "en": "Question the case"},
    "ask_hint": {
        "fr": (
            "Posez une question à l'écrit ou à la voix ; la réponse vient "
            "uniquement du dossier scellé."
        ),
        "en": "Ask in writing or by voice; the answer comes only from the sealed case.",
    },
    "ask_question": {"fr": "Votre question", "en": "Your question"},
    "ask_record": {"fr": "Ou poser la question au micro", "en": "Or ask with the microphone"},
    "ask_button": {"fr": "Poser la question", "en": "Ask the question"},
    "ask_transcribing": {"fr": "Écoute de la question…", "en": "Listening to the question…"},
    "ask_heard": {"fr": "Question entendue : {question}", "en": "Question heard: {question}"},
    "ask_need": {
        "fr": "Écrivez ou enregistrez d'abord une question.",
        "en": "Write or record a question first.",
    },
    "ask_running": {"fr": "Recherche dans le dossier…", "en": "Searching the case…"},
    "ask_sources": {"fr": "Sources", "en": "Sources"},
    "tab_relief": {"fr": "Relief", "en": "Relief"},
    "relief_hint": {
        "fr": (
            "Téléphone immobile, une seule lampe déplacée "
            "à 12 h, 3 h, 6 h puis 9 h (environ 45° de hauteur)."
        ),
        "en": "Still phone, one lamp moved to 12, 3, 6 then 9 o'clock (about 45° high).",
    },
    "relief_tx_label": {"fr": "Identifiant du dossier", "en": "Case identifier"},
    "case_manual": {
        "fr": "Saisir un identifiant (lien reçu)…",
        "en": "Enter an identifier (received link)…",
    },
    "case_manual_input": {
        "fr": "Identifiant du dossier (ex. tx-1a2b3c4d)",
        "en": "Case identifier (e.g. tx-1a2b3c4d)",
    },
    "case_no_description": {
        "fr": "objet pas encore photographié",
        "en": "item not photographed yet",
    },
    "case_is_code": {
        "fr": (
            "Ceci est un code défi, pas un identifiant de dossier. "
            "Choisissez le dossier dans la liste."
        ),
        "en": "This is a challenge code, not a case identifier. Pick the case from the list.",
    },
    "relief_side": {"fr": "Constat à enrichir", "en": "Constat to enrich"},
    "relief_concept": {
        "fr": "Objet à détourer, en anglais (ex. headphones)",
        "en": "Object to outline (e.g. headphones)",
    },
    "relief_uploads": {
        "fr": "Photos en lumière rasante, dans l'ordre 12 h, 3 h, 6 h, 9 h (4 à 8)",
        "en": "Raking-light photos, in 12, 3, 6, 9 o'clock order (4 to 8)",
    },
    "relief_uncalibrated": {
        "fr": "Positions de lampe inconnues (mode non calibré)",
        "en": "Unknown lamp positions (uncalibrated mode)",
    },
    "relief_button": {"fr": "Mesurer le relief", "en": "Measure the relief"},
    "relief_running": {"fr": "Mesure du relief en cours…", "en": "Measuring relief…"},
    "relief_need_photos": {
        "fr": "Déposez 4 à 8 photos du même objet.",
        "en": "Drop 4 to 8 photos of the same item.",
    },
    "relief_no_constat": {
        "fr": "Ce dossier n'a pas encore de constat {side}.",
        "en": "This case has no {side} constat yet.",
    },
    "relief_elapsed": {"fr": "Calcul CPU : {s} s.", "en": "CPU compute: {s} s."},
    "relief_method_calibrated": {
        "fr": "Méthode : lumière rasante calibrée (12 h, 3 h, 6 h, 9 h).",
        "en": "Method: calibrated raking light (12, 3, 6, 9 o'clock).",
    },
    "relief_method_uncalibrated": {
        "fr": "Méthode : lumière rasante non calibrée (Hayakawa).",
        "en": "Method: uncalibrated raking light (Hayakawa).",
    },
    "relief_mask_sam": {"fr": "Masque : SAM 3.1.", "en": "Mask: SAM 3.1."},
    "relief_mask_threshold": {
        "fr": "Masque : seuillage simple (SAM indisponible).",
        "en": "Mask: plain threshold (SAM unavailable).",
    },
    "relief_ok": {
        "fr": "Relief mesuré sur {pct} % du masque.",
        "en": "Relief measured on {pct}% of the mask.",
    },
    "relief_flat": {
        "fr": "RELIEF PLAT — possible photo d'écran ou impression.",
        "en": "FLAT RELIEF — possible screen photo or print.",
    },
    "relief_map": {
        "fr": "Carte de relief (bleu : creux, rouge : bosses)",
        "en": "Relief map (blue: hollows, red: bumps)",
    },
    "relief_normals": {"fr": "Carte des normales", "en": "Normals map"},
    "relief_reading": {"fr": "Lecture du relief", "en": "Relief reading"},
    "relief_features_none": {
        "fr": "Aucun creux ni bosse décrit.",
        "en": "No hollows or bumps described.",
    },
    "relief_col_shape": {"fr": "Forme", "en": "Shape"},
    "relief_col_zone": {"fr": "Zone", "en": "Zone"},
    "relief_col_defect": {"fr": "Défaut lié", "en": "Linked flaw"},
    "relief_attach": {
        "fr": "Sceller dans le constat {side}",
        "en": "Seal into the {side} constat",
    },
    "relief_attached": {
        "fr": "Relief scellé dans le constat {side}.",
        "en": "Relief sealed into the {side} constat.",
    },
    "relief_seal_note": {
        "fr": "Le relief rejoint le constat ; il sera couvert par le sceau du prochain certificat.",
        "en": "The relief joins the constat; it will be covered by the next certificate seal.",
    },
    "relief_already": {
        "fr": "Ce constat contient déjà un relief (relancez pour le remplacer).",
        "en": "This constat already holds a relief (run again to replace it).",
    },
    "relief_invalid": {
        "fr": "Rapport de relief illisible : relancez la mesure.",
        "en": "Unreadable relief report: run the measurement again.",
    },
}

VERDICTS: dict[str, dict[str, str]] = {
    "CONFORME": {"fr": "CONFORME", "en": "MATCH"},
    "ENDOMMAGÉ": {"fr": "ENDOMMAGÉ", "en": "DAMAGED IN TRANSIT"},
    "À VÉRIFIER": {"fr": "À VÉRIFIER", "en": "NEEDS REVIEW"},
    "NON CONFORME": {"fr": "NON CONFORME", "en": "MISMATCH"},
}


def t(key: str, lang: str) -> str:
    """Return the UI string for a key, falling back to French then the key."""
    entry = STRINGS.get(key, {})
    return entry.get(lang) or entry.get("fr") or key


def verdict_label(verdict: str, lang: str) -> str:
    """Return the verdict label in the user language."""
    entry = VERDICTS.get(verdict, {})
    return entry.get(lang) or entry.get("fr") or verdict
