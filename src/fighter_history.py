#!/usr/bin/env python3
"""
Muestra el historial de un peleador con su ELO, el ELO del rival al
momento de la pelea y el cambio de ELO de cada uno al finalizar.

Replica exactamente la logica de Elo de `fighter_engine.py`:
  - K variable por experiencia (96/64/40/24)
  - Decaimiento por inactividad >1 año
  - Sin ELO update en Draw/NC/DQ

Input interactivo por terminal con autocompletado (como predict.py):
  - Escribe parte del nombre y lista coincidencias
  - Selecciona por numero
  - Sin parametros CLI

  python src/fighter_history.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIGHTS_PATH, ELO_INITIAL
from fighter_engine import (
    FightStateEngine,
    classify_method,
    get_k_factor,
    elo_expected,
    elo_update,
)


def load_fights(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_history(target: str, fights: list):
    """Recorre peleas cronologicamente y extrae el historial de `target`.

    Delega el loop cronológico, Elo decay y post-fight updates a
    ``FightStateEngine`` (``cutoff=None`` para mostrar historial completo
    pre-2012 también). El cálculo de ELO pre/post/expected replica
    exactamente la lógica de ``fighter_engine`` (K variable, decay, Elo).
    """
    target_lower = target.lower()
    # cutoff=None -> incluye toda la historia, igual que el comportamiento previo
    engine = FightStateEngine(fights, cutoff=None)
    history = []

    for fight in engine:
        f1, f2 = fight["fighter_1"], fight["fighter_2"]
        date = fight["_parsed_date"]

        # Pre-fight Elo (con decay ya aplicado por el engine)
        f1_elo_before = engine.state[f1]["elo"]
        f2_elo_before = engine.state[f2]["elo"]

        is_win_loss, win_side, finish_type = classify_method(
            fight["method"], fight["winner"], f1, f2
        )

        if is_win_loss:
            f1_fights_after = engine.state[f1]["total_fights"] + 1
            f2_fights_after = engine.state[f2]["total_fights"] + 1
            k_a = get_k_factor(f1_fights_after)
            k_b = get_k_factor(f2_fights_after)
            score_a = 1.0 if win_side == 1 else 0.0
            f1_elo_after, f2_elo_after = elo_update(f1_elo_before, f2_elo_before, score_a, k_a=k_a, k_b=k_b)
        else:
            f1_elo_after = f1_elo_before
            f2_elo_after = f2_elo_before

        exp_f1 = elo_expected(f1_elo_before, f2_elo_before)
        exp_f2 = 1.0 - exp_f1

        is_f1_target = f1.lower() == target_lower
        is_f2_target = f2.lower() == target_lower
        if is_f1_target or is_f2_target:
            if is_f1_target:
                elo_before = f1_elo_before
                elo_after = f1_elo_after
                opp_name = f2
                opp_before = f2_elo_before
                opp_after = f2_elo_after
                expected = exp_f1
                if not is_win_loss:
                    result = fight["winner"]
                else:
                    result = "W" if win_side == 1 else "L"
            else:
                elo_before = f2_elo_before
                elo_after = f2_elo_after
                opp_name = f1
                opp_before = f1_elo_before
                opp_after = f1_elo_after
                expected = exp_f2
                if not is_win_loss:
                    result = fight["winner"]
                else:
                    result = "W" if win_side == 2 else "L"

            history.append({
                "event_date": fight["event_date"],
                "_parsed_date": date,
                "event_name": fight.get("event_name", ""),
                "category": fight.get("category", ""),
                "opponent": opp_name,
                "result": result,
                "method": fight.get("method", ""),
                "round": fight.get("round", 0),
                "time": fight.get("time", ""),
                "winner": fight.get("winner", ""),
                "elo_before": round(elo_before, 2),
                "elo_after": round(elo_after, 2),
                "delta": round(elo_after - elo_before, 2),
                "opp_elo_before": round(opp_before, 2),
                "opp_elo_after": round(opp_after, 2),
                "opp_delta": round(opp_after - opp_before, 2),
                "expected_win_prob": round(expected, 4),
            })

        # El engine aplica update_state + Elo update al reanudar el generador
        # (estado completo: stats, recent_fights, opp_elo, etc.)

    return history, engine.state


# --- autocomplete interactivo (igual que predict.py) ---

def find_fighter(query: str, all_names: set[str]) -> list[str]:
    q = query.lower().strip()
    matches = [n for n in all_names if q in n.lower()]
    return sorted(matches)


def select_fighter(prompt: str, all_names: set[str]) -> str | None:
    """Loop interactivo con autocomplete. Retorna nombre o None si usuario escribe 'salir'."""
    while True:
        try:
            query = input(f"\n{prompt} (o 'salir' para terminar): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if not query:
            print("  Ingresa un nombre.")
            continue
        if query.lower() in ("salir", "exit", "quit", "q"):
            return None

        matches = find_fighter(query, all_names)

        if len(matches) == 0:
            print(f"  No se encontraron peleadores con '{query}'.")
            continue

        if len(matches) == 1:
            print(f"  Seleccionado: {matches[0]}")
            return matches[0]

        print(f"\n  Multiples coincidencias para '{query}':")
        for i, name in enumerate(matches, 1):
            print(f"    {i}. {name}")
        print(f"    {len(matches) + 1}. Buscar otro nombre")

        try:
            choice = input("\n  Elige numero: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if not choice:
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(matches):
                return matches[idx]
            elif idx == len(matches):
                continue
            else:
                print("  Numero fuera de rango.")
        except ValueError:
            # si escribe texto de nuevo, usarlo como nueva query
            # re-evaluar como query directa
            retry_matches = find_fighter(choice, all_names)
            if len(retry_matches) == 1:
                print(f"  Seleccionado: {retry_matches[0]}")
                return retry_matches[0]
            elif len(retry_matches) > 1:
                # mostrar de nuevo
                print(f"\n  Multiples coincidencias para '{choice}':")
                for i, name in enumerate(retry_matches, 1):
                    print(f"    {i}. {name}")
                continue
            else:
                print("  Entrada no valida.")
                continue


def print_table(history: list, target: str):
    if not history:
        print(f"Sin peleas encontradas para '{target}'.")
        return

    print(f"\nHistorial de {target}  ({len(history)} peleas)")
    print(f"ELO inicial = {ELO_INITIAL}  |  K = 96/64/40/24 segun experiencia  |  decay >1 año inactivo")
    print("-" * 140)
    header = (
        f"{'#':>3}  {'Fecha':<10}  {'Evento':<30}  {'Rival':<22}  "
        f"{'Res':<3} {'Metodo':<14} {'R':>1}  "
        f"{'ELO':>7} {'ELO_riv':>8} {'P(win)':>6}  "
        f"{'Delta':>7} {'ELO_post':>8}  {'D_riv':>7} {'ELO_riv_post':>12}"
    )
    print(header)
    print("-" * 140)

    wins = losses = draws = ncs = 0
    for i, h in enumerate(history, 1):
        event = h["event_name"][:30]
        rival = h["opponent"][:22]
        method = h["method"][:14]
        delta_str = f"{h['delta']:+.1f}"
        opp_delta_str = f"{h['opp_delta']:+.1f}"
        res = h["result"]
        if res == "W":
            wins += 1
        elif res == "L":
            losses += 1
        elif res == "Draw":
            draws += 1
        elif res == "No Contest":
            ncs += 1

        print(
            f"{i:>3}  {h['event_date']:<10}  {event:<30}  {rival:<22}  "
            f"{res:<3} {method:<14} {h['round']:>1}  "
            f"{h['elo_before']:>7.1f} {h['opp_elo_before']:>8.1f} {h['expected_win_prob']:>6.3f}  "
            f"{delta_str:>7} {h['elo_after']:>8.1f}  {opp_delta_str:>7} {h['opp_elo_after']:>12.1f}"
        )

    print("-" * 140)
    last = history[-1]
    print(f"Peleas: {len(history)}  |  W-L-D/NC: {wins}-{losses}-{draws+ ncs} (D={draws} NC={ncs})")
    print(f"ELO actual: {last['elo_after']:.1f}  |  Pico: {max(h['elo_after'] for h in history):.1f}  |  Valle: {min(h['elo_after'] for h in history):.1f}")
    streak = 0
    for h in reversed(history):
        if h["result"] == "W":
            if streak >= 0:
                streak += 1
            else:
                break
        elif h["result"] == "L":
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break
    if streak > 0:
        print(f"Racha actual: {streak}W")
    elif streak < 0:
        print(f"Racha actual: {abs(streak)}L")
    else:
        print("Racha actual: -")
    print()


def main():
    print("=" * 60)
    print("  HISTORIAL ELO DE PELEADOR")
    print("=" * 60)

    print("\nCargando peleas...")
    fights = load_fights(FIGHTS_PATH)
    print(f"  {len(fights)} peleas cargadas")

    all_names: set[str] = set()
    for f in fights:
        all_names.add(f["fighter_1"])
        all_names.add(f["fighter_2"])
    print(f"  {len(all_names)} peleadores en fights.json")

    while True:
        fighter = select_fighter("Ingresa nombre del peleador", all_names)
        if fighter is None:
            print("\nSaliendo.")
            break

        history, _ = build_history(fighter, fights)

        if not history:
            print(f"'{fighter}' sin peleas en fights.json (inesperado).")
            continue

        print_table(history, fighter)

        # loop continuo sin preguntar; el proximo select_fighter permite salir con 'salir'


if __name__ == "__main__":
    main()
