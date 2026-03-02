# soul_engine.py
# pip install numpy uvicorn fastapi pydantic transformers accelerate
# FIX FOR VAST.AI 
#netstat -tulpn | grep 8384
#mv /opt/portal-aio/caddy_manager/caddy /opt/portal-aio/caddy_manager/caddy.BLOQUEADO
#kill -9 PROCESS NUMBER 
import asyncio
import os
import torch
import numpy as np
import uvicorn
import logging
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
import json
from pathlib import Path

# --- CONFIGURAÇÃO ---
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-8B")
PORT = int(os.getenv("PORT", "54213"))
VECTOR_DIR = "./vectors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIGURAÇÃO OTIMIZADA PARA 3090
MAX_CONCURRENT_GPU = 1  # 3090 só aguenta 1 por vez com 14B
LOAD_IN_8BIT = False  # Sem quantização, usa offloading CPU/GPU

gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPU)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SoulEngine")


# --- ESTADO GLOBAL ---
class EngineState:
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    vectors: Dict[str, torch.Tensor] = {}
    active_hooks: List[Any] = []
    initialization_complete: bool = False


state = EngineState()


# --- SCHEMAS ---
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024  # REDUZIDO: 1024 ao invés de 2048
    temperature: float = 0.6
    top_p: float = 0.95
    vector_name: Optional[str] = None
    intensity: float = 0.0
    layer_idx: int = 16


class EmbeddingRequest(BaseModel):
    text: str
    layer_idx: int = -1


class CalibrationRequest(BaseModel):
    concept_name: str
    positive_samples: List[str]
    negative_samples: List[str]
    layer_idx: int = 16


# --- FASTAPI APP ---
app = FastAPI(
    title="Aura Soul Engine (RTX 3090 Optimized)",
    description="LLM com Steering Vectors - Otimizado para 24GB VRAM",
    version="3.2"
)

app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- FUNÇÕES DE EXTRAÇÃO DE HIDDEN STATES ---
def get_hidden_states(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        texts: List[str],
        layer_idx: int = -1
) -> List[np.ndarray]:
    """
    Extrai hidden states de uma lista de textos.
    Retorna: List[np.ndarray] - Um vetor por texto
    """
    hidden_states_list = []

    # Hook para capturar hidden states
    model_layers = getattr(model, 'model', model).layers  # type: ignore[attr-defined]
    target_layer = model_layers[layer_idx] if layer_idx >= 0 else model_layers[-1]
    captured_states: List[torch.Tensor] = []

    def hook_fn(module: Any, input: Any, output: Any) -> None:
        # Output pode ser tupla (hidden_states, ...)
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Pega último token: [batch, seq_len, hidden_dim]
        last_token_state = hidden[:, -1, :].detach().cpu()
        # Converte para float32 (compatibilidade com NumPy)
        if last_token_state.dtype in [torch.bfloat16, torch.float16]:
            last_token_state = last_token_state.to(torch.float32)
        captured_states.append(last_token_state)

    hook_handle = target_layer.register_forward_hook(hook_fn)

    try:
        for text in texts:
            # Tokeniza com limite menor para economizar VRAM
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                _ = model(**inputs)

            # Pega o estado capturado
            if captured_states:
                state_vector = captured_states[-1].squeeze(0).numpy()
                hidden_states_list.append(state_vector)

            # Limpa cache da GPU após cada texto
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    finally:
        hook_handle.remove()

    return hidden_states_list


# --- EXTRAÇÃO DE VETORES DE CONTROLE (DiffMean) ---
def extract_control_vector(
        positive_samples: List[str],
        negative_samples: List[str],
        layer_idx: int = 16
) -> np.ndarray:
    """
    Extrai vetor de controle usando método DiffMean (similar ao EasySteer).
    """
    logger.info(f"🧪 Extraindo vetor na camada {layer_idx}...")

    if state.model is None or state.tokenizer is None:
        raise RuntimeError("Model or tokenizer not initialized")

    # Extrai hidden states
    pos_states = get_hidden_states(state.model, state.tokenizer, positive_samples, layer_idx)
    neg_states = get_hidden_states(state.model, state.tokenizer, negative_samples, layer_idx)

    # Converte para arrays
    X_pos = np.array(pos_states)
    X_neg = np.array(neg_states)

    # DiffMean: média das diferenças
    pos_mean = X_pos.mean(axis=0)
    neg_mean = X_neg.mean(axis=0)

    control_vector = pos_mean - neg_mean

    # Normaliza
    norm = np.linalg.norm(control_vector)
    if norm > 0:
        control_vector = control_vector / norm

    logger.info(f"✅ Vetor extraído com dimensão {control_vector.shape[0]}")
    return control_vector


# --- STEERING HOOK ---
def create_steering_hook(steering_vector: torch.Tensor, intensity: float):
    """
    Cria um hook que injeta o steering vector nas hidden states.
    """

    def hook_fn(module: Any, input: Any, output: Any) -> Union[torch.Tensor, tuple]:
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()

        # Prepara o vetor para injeção
        steer_vec = steering_vector.to(hidden_states.device).to(hidden_states.dtype)

        # Injeta em todos os tokens da sequência
        batch_size, seq_len, hidden_dim = hidden_states.shape
        steer_vec_expanded = steer_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

        # Adiciona o vetor escalado
        hidden_states = hidden_states + (intensity * steer_vec_expanded)

        if rest:
            return (hidden_states,) + rest
        return hidden_states

    return hook_fn


def clear_hooks() -> None:
    """Remove todos os hooks ativos."""
    for hook in state.active_hooks:
        hook.remove()
    state.active_hooks.clear()


# --- CARREGAR VETORES SALVOS ---
def load_vectors() -> None:
    """Carrega vetores salvos do disco."""
    os.makedirs(VECTOR_DIR, exist_ok=True)

    for file in Path(VECTOR_DIR).glob("*.npy"):
        concept_name = file.stem
        vector = np.load(file)
        state.vectors[concept_name] = torch.tensor(vector, dtype=torch.float32)
        logger.info(f"📦 Vetor carregado: {concept_name}")


# --- STARTUP ---
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=" * 60)
    logger.info(f"🚀 Inicializando Soul Engine em {DEVICE}")
    logger.info(f"📦 Carregando modelo: {MODEL_ID}")
    logger.info(f"⚙️  Modo: FP16 com CPU Offloading (RTX 3090 Optimized)")

    try:
        # Carrega tokenizer
        state.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )

        # Carrega modelo
        if DEVICE == "cuda":
            # FP16 com offloading automático para CPU/RAM quando necessário
            logger.info("🔧 Carregando com FP16 + CPU offloading...")

            state.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",  # Distribui automaticamente entre GPU e CPU
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                max_memory={0: "22GB", "cpu": "24GB"}  # GPU: 22GB, CPU: 24GB
            )
        else:
            # CPU fallback
            state.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            state.model = state.model.to(DEVICE)

        # Carrega vetores existentes
        load_vectors()

        # Obtém número de camadas
        model_layers = getattr(state.model, 'model', state.model).layers  # type: ignore[attr-defined]
        num_layers = len(model_layers)

        # Limpa cache inicial
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            logger.info(f"📊 VRAM: {allocated:.2f}GB alocada / {reserved:.2f}GB reservada")

        logger.info("=" * 60)
        logger.info(f"✅ Soul Engine pronto!")
        logger.info(f"🎯 Dispositivo: {DEVICE}")
        logger.info(f"🧠 Camadas: {num_layers}")
        logger.info(f"📦 Vetores carregados: {len(state.vectors)}")
        logger.info(f"⚡ Concorrência GPU: {MAX_CONCURRENT_GPU}")
        logger.info("=" * 60)

        state.initialization_complete = True

    except Exception as e:
        logger.error(f"❌ ERRO CRÍTICO na inicialização: {e}", exc_info=True)
        state.initialization_complete = False
        raise


# --- HEALTH CHECK ---
@app.get("/")
async def root() -> Dict[str, Any]:
    """Health check endpoint."""
    if not state.initialization_complete:
        raise HTTPException(status_code=503, detail="Model still initializing")

    model_layers = getattr(state.model, 'model', state.model).layers  # type: ignore[attr-defined]
    num_layers = len(model_layers)

    vram_info = {}
    if DEVICE == "cuda":
        vram_info = {
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
        }

    return {
        "status": "ready",
        "model": MODEL_ID,
        "device": DEVICE,
        "quantization": "FP16 + CPU Offload",
        "vectors_loaded": list(state.vectors.keys()),
        "total_layers": num_layers,
        "max_concurrent_gpu": MAX_CONCURRENT_GPU,
        **vram_info
    }


@app.post("/embed")
async def embed_geometry(request: EmbeddingRequest) -> Dict[str, Any]:
    """
    Retorna o vetor latente (geometria) do texto.
    """
    if not state.initialization_complete:
        raise HTTPException(status_code=503, detail="Model still initializing, please wait")

    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        async with asyncio.timeout(30):
            async with gpu_semaphore:
                try:
                    layer_idx = request.layer_idx if request.layer_idx >= 0 else -1

                    hidden_states = get_hidden_states(
                        state.model,
                        state.tokenizer,
                        [request.text],
                        layer_idx
                    )

                    vector = hidden_states[0].tolist()

                    return {
                        "vector": vector,
                        "dim": len(vector),
                        "model": MODEL_ID,
                        "layer": layer_idx
                    }

                except Exception as e:
                    logger.error(f"❌ Erro na extração: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
                finally:
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

    except asyncio.TimeoutError:
        logger.error("⏱️ Timeout ao processar embedding")
        raise HTTPException(status_code=504, detail="Request timeout - GPU busy")


@app.post("/generate_with_soul")
async def generate_with_soul(request: GenerateRequest) -> Dict[str, Any]:
    """
    Gera texto com injeção de vetor (steering) opcional.
    """
    if not state.initialization_complete:
        raise HTTPException(status_code=503, detail="Model still initializing, please wait")

    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        async with asyncio.timeout(120):
            async with gpu_semaphore:
                try:
                    clear_hooks()

                    # Aplica steering se solicitado
                    if request.vector_name and request.vector_name in state.vectors:
                        if request.intensity != 0:
                            logger.info(
                                f"💉 Injetando '{request.vector_name}' (Int: {request.intensity}) na camada {request.layer_idx}"
                            )

                            model_layers = getattr(state.model, 'model',
                                                   state.model).layers  # type: ignore[attr-defined]
                            num_layers = len(model_layers)
                            if not (0 <= request.layer_idx < num_layers):
                                # Se a camada for errada, não quebra a API, apenas ajusta para o limite
                                request.layer_idx = min(request.layer_idx, num_layers - 1)
                                logger.warning(f"Ajustando layer_idx para o limite seguro: {request.layer_idx}")

                            target_layer = model_layers[request.layer_idx]
                            steering_vec = state.vectors[request.vector_name]
                            hook_fn = create_steering_hook(steering_vec, request.intensity)
                            hook_handle = target_layer.register_forward_hook(hook_fn)
                            state.active_hooks.append(hook_handle)

                    elif request.vector_name and request.vector_name not in state.vectors:
                        # [V5 FIX] Resiliência Robusta.
                        # Em vez de retornar Error 400 e quebrar a conversa do usuário,
                        # logamos o aviso e continuamos a geração SEM o hormônio.
                        logger.error(
                            f"⚠️ VETOR FANTASMA: '{request.vector_name}' não carregado. Ignorando steering."
                        )

                    # Prepara prompt
                    if "<|im_start|>" not in request.prompt:
                        messages = [
                            {"role": "system", "content": "You are Aura, an advanced AI assistant."},
                            {"role": "user", "content": request.prompt}
                        ]
                        formatted_prompt = state.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                    else:
                        formatted_prompt = request.prompt

                    # Tokeniza
                    inputs = state.tokenizer(formatted_prompt, return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                    # Configuração de geração
                    gen_config: Dict[str, Any] = {
                        "max_new_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "do_sample": request.temperature > 0,
                        "pad_token_id": state.tokenizer.eos_token_id,
                    }

                    if hasattr(state.tokenizer, 'im_end_id'):
                        gen_config["eos_token_id"] = [
                            state.tokenizer.eos_token_id,
                            state.tokenizer.im_end_id  # type: ignore[attr-defined]
                        ]

                    # Gera
                    with torch.no_grad():
                        outputs = state.model.generate(**inputs, **gen_config)

                    # Decodifica
                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                    text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)

                    return {
                        "text": text,
                        "model": MODEL_ID,
                        "tokens_generated": len(generated_ids)
                    }

                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"❌ Erro na geração: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))

                finally:
                    clear_hooks()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

    except asyncio.TimeoutError:
        logger.error("⏱️ Timeout ao gerar texto")
        raise HTTPException(status_code=504, detail="Generation timeout - GPU busy")


@app.post("/calibrate")
async def calibrate(request: CalibrationRequest) -> Dict[str, Any]:
    """
    Cria novos vetores (Conceitos/Hormônios) usando DiffMean.
    """
    if not state.initialization_complete:
        raise HTTPException(status_code=503, detail="Model still initializing, please wait")

    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        async with asyncio.timeout(300):
            async with gpu_semaphore:
                try:
                    if len(request.positive_samples) < 2 or len(request.negative_samples) < 2:
                        raise HTTPException(
                            status_code=400,
                            detail="Mínimo 2 amostras positivas e 2 negativas"
                        )

                    model_layers = getattr(state.model, 'model', state.model).layers  # type: ignore[attr-defined]
                    num_layers = len(model_layers)
                    if not (0 <= request.layer_idx < num_layers):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Layer {request.layer_idx} inválido. Máx: {num_layers - 1}"
                        )

                    logger.info(f"🧪 Calibrando conceito: '{request.concept_name}'")

                    control_vector = extract_control_vector(
                        request.positive_samples,
                        request.negative_samples,
                        request.layer_idx
                    )

                    os.makedirs(VECTOR_DIR, exist_ok=True)
                    filepath = os.path.join(VECTOR_DIR, f"{request.concept_name}.npy")
                    np.save(filepath, control_vector)

                    state.vectors[request.concept_name] = torch.tensor(control_vector, dtype=torch.float32)

                    logger.info(f"✅ Conceito '{request.concept_name}' criado com sucesso!")

                    return {
                        "status": "success",
                        "concept": request.concept_name,
                        "layer_used": request.layer_idx,
                        "vector_path": filepath,
                        "vector_dim": len(control_vector)
                    }

                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"❌ Erro na calibração: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))
                finally:
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

    except asyncio.TimeoutError:
        logger.error("⏱️ Timeout ao calibrar conceito")
        raise HTTPException(status_code=504, detail="Calibration timeout")


@app.get("/concepts")
async def list_concepts() -> Dict[str, Any]:
    """Lista todos os vetores disponíveis."""
    return {
        "concepts": list(state.vectors.keys()),
        "total": len(state.vectors)
    }


@app.delete("/concepts/{concept_name}")
async def delete_concept(concept_name: str) -> Dict[str, str]:
    """Remove um conceito."""
    if concept_name not in state.vectors:
        raise HTTPException(status_code=404, detail=f"Conceito '{concept_name}' não encontrado")

    del state.vectors[concept_name]

    filepath = os.path.join(VECTOR_DIR, f"{concept_name}.npy")
    if os.path.exists(filepath):
        os.remove(filepath)

    return {"status": "deleted", "concept": concept_name}


# --- MAIN ---
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1111,
        log_level="info",
        limit_concurrency=50,  # Reduzido: apenas 1 GPU request por vez
        timeout_keep_alive=300,
        backlog=2048
    )
