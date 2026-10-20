__all__ = ["PRETRAINED"]

PRETRAINED = {
    "hovernet": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "csmp-hub/hovernet-histo-hgsc-nuc-v1",
            "filename": "hovernet_hgsc_v1_efficientnet_b5.safetensors",
        },
    },
    "cellpose": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "csmp-hub/cellpose-histo-hgsc-nuc-v1",
            "filename": "cellpose_hgsc_v1_efficientnet_b5.safetensors",
        },
    },
    "cellvit": {
        "hgsc_v1_samvit_base_patch16": {
            "repo_id": "csmp-hub/cellvit-histo-hgsc-nuc-v1",
            "filename": "cellvit_hgsc_v1_efficientnet_b5.safetensors",
        },
    },
    "stardist": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "csmp-hub/stardist-histo-hgsc-nuc-v1",
            "filename": "stardist_hgsc_v1_efficientnet_b5.safetensors",
        },
    },
    "instanseg": {},
}

# Keep the original checkpoint name working with the existing Hub file.
PRETRAINED["cellvit"]["hgsc_v1_efficientnet_b5"] = PRETRAINED["cellvit"][
    "hgsc_v1_samvit_base_patch16"
]
