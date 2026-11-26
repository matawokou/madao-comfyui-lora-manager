from typing import Optional

# Base model mapping based on version string
BASE_MODEL_MAPPING = {
    "sd_1.5": "SD 1.5",
    "sd-v1-5": "SD 1.5",
    "sd-v2-1": "SD 2.1",
    "sdxl": "SDXL 1.0",
    "sd-v2": "SD 2.0",
    "flux1": "Flux.1 D",
    "flux.1 d": "Flux.1 D",
    "illustrious": "Illustrious",
    "il": "Illustrious",
    "pony": "Pony",
    "Hunyuan Video": "Hunyuan Video",
    # Krea 2.  MuseAI/ModelScope-trained LoRAs write a plain
    # ``base_model = "krea/Krea-2-Turbo"`` header instead of the A1111-style
    # ``ss_base_model_version``, so these spellings have to be recognised here.
    # Deliberately NOT a bare "krea" key: "flux.1 krea" is a Flux.1 variant,
    # not Krea 2, and a bare key would swallow it.
    "krea-2": "Krea 2",
    "krea2": "Krea 2",
    "krea 2": "Krea 2",
    "krea_2": "Krea 2"
}

def determine_base_model(version_string: Optional[str]) -> str:
    """Determine base model from version string in safetensors metadata"""
    if not version_string:
        return "Unknown"
    
    version_lower = version_string.lower()
    for key, value in BASE_MODEL_MAPPING.items():
        if key in version_lower:
            return value
    
    # TODO: Add more base model mappings
    return version_string 