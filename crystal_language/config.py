"""Configuration dataclasses for the crystal language route."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    n_layer: int = 8
    n_encoder_layer: int = 0
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    block_size: int = 1024
    # "v1": LayerNorm + learned positions + GELU (legacy checkpoints).
    # "v2": RMSNorm + RoPE + SwiGLU + dense spectra prefix encoder.
    arch: str = "v1"
    spectra_slots: int = 16
    spectra_grid: int = 1501  # 5-80 deg @ 0.05 deg
    # "cnn": fixed 2θ raster; "di": (d, I) peak-list transformer.
    spectra_encoder: str = "cnn"
    spectra_max_peaks: int = 128
    spectra_encoder_layers: int = 3
    rope_base: float = 10000.0
    # V2 conditioning: per-block AdaLN (FiLM) modulation from a pooled spectra
    # condition vector. Off by default so legacy v2 checkpoints keep loading.
    cond_adanorm: bool = False
    # Train-time condition dropout for classifier-free guidance (drop the whole
    # condition vector to the learned null embedding with this probability).
    cond_drop: float = 0.1
    # V2 geometry: add a Fourier embedding of each coordinate / lattice bin
    # token's numeric value to its token embedding, so adjacent bins are close
    # in embedding space (discrete bins otherwise carry no metric structure).
    coord_value_emb: bool = False
    coord_value_freqs: int = 16


@dataclass
class DataConfig:
    data_dir: str = "data_prep/outputs/crystal_language_mp_wbm"
    condition_set: str = "mix"
    use_lattice: str = "mix"  # "always", "never", or "mix"
    target_mode: str = "full"  # "full" or "sites"
    block_size: int = 1024
    max_records: int | None = None
    # Optional n_sites window for size-window specialists. None disables that
    # bound. min_sites=21 with max_sites=None is the historical >20 expert;
    # min_sites=13 and max_sites=20 is the SI-band 13–20 expert.
    min_sites: int | None = None
    max_sites: int | None = None
    rep: str = "v1"  # sequence representation: "v1" char tokens, "v2" bin tokens
    spectra_slots: int = 16
    spectra_grid_min: float = 5.0
    spectra_grid_max: float = 80.0
    spectra_grid_step: float = 0.05
    spectra_sigma: float = 0.15
    spectra_encoder: str = "cnn"
    spectra_max_peaks: int = 128
    spectra_wavelength: float = 1.54184
    noise_profile: str = "none"
    # When --use-lattice=mix, probability of a lattice-free prompt.
    lattice_free_prob: float = 0.5
    mix_condition_sets: list[str] = field(
        default_factory=lambda: [
            "pxrd",
            "pxrd_element",
            "pxrd_xrf",
            "pxrd_neutron",
            "pxrd_neutron_xrf",
            "pxrd_neutron_xrf_xps",
            "pxrd_formula",
        ]
    )
    mix_condition_weights: list[float] | None = None


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 3.0e-4
    weight_decay: float = 0.1
    max_steps: int = 20000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 2000
    num_workers: int = 4
    seed: int = 42
    out_dir: str = "crystal_language/runs/mp_wbm_v1"
    device: str = "cuda"
    amp: str = "none"  # "none", "fp16", or "bf16"
