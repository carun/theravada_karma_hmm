import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class PathStage(Enum):
    ORDINARY = "ordinary"
    STREAM_ENTRY = "stream_entry"
    ONCE_RETURNER = "once_returner"
    NON_RETURNER = "non_returner"
    ARAHANT = "arahant"

class RebirthRealm(Enum):
    HELL = "hell"
    ANIMAL = "animal"
    GHOST = "ghost"
    HUMAN = "human"
    DEVA = "deva"
    BRAHMA = "brahma"

class MeditationType(Enum):
    SAMATHA = "samatha"
    VIPASSANA = "vipassana"
    SATIPATTHANA = "satipatthana"
    ANAPANASATI = "anapanasati"
    METTA = "metta"
    KASINA = "kasina"

class RipeningType(Enum):
    IMMEDIATE = "immediate"
    NEXT_LIFE = "next_life"
    LATER_LIVES = "later_lives"
    INDEFINITE = "indefinite"

@dataclass
class RebirthCircumstance:
    realm: RebirthRealm
    family_conditions: str
    physical_conditions: str
    mental_conditions: str
    spiritual_conditions: str
    environmental_conditions: str
    primary_kilesas: List[str]
    life_span_range: Tuple[int, int]
    karmic_momentum: float

@dataclass
class MeditationPractice:
    practice_type: MeditationType
    daily_duration: float
    consistency: float
    quality: float
    years_practiced: float
    teacher_guidance: float
    retreat_hours: float

    def get_kilesa_suppression_factor(self) -> float:
        base_factor = (self.daily_duration * self.consistency * self.quality) / 24
        experience_bonus = min(0.5, self.years_practiced * 0.05)
        retreat_bonus = min(0.3, self.retreat_hours / 1000)
        guidance_bonus = self.teacher_guidance * 0.2
        return min(0.95, base_factor + experience_bonus + retreat_bonus + guidance_bonus)

    def get_wholesome_cultivation_factor(self) -> float:
        return self.get_kilesa_suppression_factor() * 0.8

@dataclass
class RipeningTrigger:
    situational_factors: Dict[str, float]
    mental_state_factors: Dict[str, float]
    karmic_readiness: float
    astrological_factors: float
    social_context: Dict[str, float]
    health_factors: Dict[str, float]

@dataclass
class KarmicSeed:
    kilesa_type: str
    intensity: float
    creation_time: int
    wholesome: bool
    ripening_probabilities: Dict[RipeningType, float]
    decay_rate: float
    object_weight: float = 1.0
    trigger_conditions: Optional[RipeningTrigger] = None
    ripening_history: List[Dict] = field(default_factory=list)

    def current_strength(self, current_time: int) -> float:
        time_diff = current_time - self.creation_time
        return self.intensity * np.exp(-self.decay_rate * time_diff)

    def calculate_ripening_probability(self, current_context: Dict, meditation_suppression: float = 0.0) -> float:
        base_prob = 0.01
        current_str = self.current_strength(current_context.get('time', 0))
        strength_factor = current_str / max(self.intensity, 0.01)
        context_match = 1.0
        if self.trigger_conditions:
            context_match = self._calculate_context_match(current_context)
        suppression_factor = 1.0 - meditation_suppression
        timing_factor = self._calculate_timing_factor(current_context.get('time', 0))
        return base_prob * strength_factor * context_match * suppression_factor * timing_factor

    def _calculate_context_match(self, current_context: Dict) -> float:
        if not self.trigger_conditions:
            return 1.0
        match_score = 0.0
        total_weight = 0.0
        for factor, weight in self.trigger_conditions.situational_factors.items():
            if factor in current_context:
                match_score += weight * current_context[factor]
                total_weight += weight
        for factor, weight in self.trigger_conditions.mental_state_factors.items():
            if factor in current_context:
                match_score += weight * current_context[factor]
                total_weight += weight
        return match_score / max(total_weight, 1.0) if total_weight > 0 else 1.0

    def _calculate_timing_factor(self, current_time: int) -> float:
        time_cycle = (current_time % 365) / 365.0
        base_timing = 0.8 + 0.4 * np.sin(2 * np.pi * time_cycle)
        seed_age = current_time - self.creation_time
        if seed_age < 10:
            age_factor = 1.2
        elif seed_age < 100:
            age_factor = 1.0
        else:
            age_factor = 0.8 + 0.2 * np.exp(-seed_age / 1000)
        return base_timing * age_factor

@dataclass
class KilesamState:
    # Core 10 Abhidhamma Defilements
    greed: float = 0.0
    hatred: float = 0.0
    delusion: float = 0.0
    conceit: float = 0.0
    wrong_view: float = 0.0
    doubt: float = 0.0
    mental_torpor: float = 0.0
    restlessness: float = 0.0
    shamelessness: float = 0.0
    recklessness: float = 0.0

    # The 10 Fetters
    personality_view: float = 0.0
    skeptical_doubt: float = 0.0
    rites_rituals: float = 0.0
    sensual_desire: float = 0.0
    ill_will: float = 0.0
    form_desire: float = 0.0
    formless_desire: float = 0.0
    pride: float = 0.0
    mental_restlessness: float = 0.0
    ignorance: float = 0.0

    # Additional Sutta Defilements
    covetousness: float = 0.0
    anger: float = 0.0
    hostility: float = 0.0
    denigration: float = 0.0
    domineering: float = 0.0
    envy: float = 0.0
    stinginess: float = 0.0
    hypocrisy: float = 0.0
    fraud: float = 0.0
    obstinacy: float = 0.0
    presumption: float = 0.0
    arrogance: float = 0.0
    vanity: float = 0.0
    negligence: float = 0.0

    # The 5 Hindrances
    sensual_craving: float = 0.0
    aversion: float = 0.0
    sloth_torpor: float = 0.0
    restlessness_worry: float = 0.0
    skeptical_doubt_hindrance: float = 0.0

    # Additional Canonical Kilesas
    worry: float = 0.0
    physical_torpor: float = 0.0
    craving: float = 0.0
    clinging: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()

    def from_dict(self, data: Dict[str, float]):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

class TheravadaKarmaHMM:
    def __init__(self):
        self.kilesa_decay_rates = {
            'hatred': 0.8, 'greed': 0.7, 'delusion': 0.02,
            'anger': 0.9, 'ill_will': 0.8, 'aversion': 0.8,
            'envy': 0.6, 'covetousness': 0.6, 'sensual_desire': 0.6, 'sensual_craving': 0.6,
            'hostility': 0.7, 'stinginess': 0.5,
            'conceit': 0.4, 'pride': 0.4, 'arrogance': 0.5, 'vanity': 0.4,
            'denigration': 0.4, 'domineering': 0.4,
            'restlessness': 0.3, 'mental_restlessness': 0.3, 'worry': 0.3,
            'restlessness_worry': 0.3, 'presumption': 0.4,
            'mental_torpor': 0.2, 'physical_torpor': 0.2, 'sloth_torpor': 0.2,
            'negligence': 0.15,
            'hypocrisy': 0.3, 'fraud': 0.35, 'obstinacy': 0.25,
            'shamelessness': 0.1, 'recklessness': 0.1,
            'wrong_view': 0.1, 'personality_view': 0.08,
            'doubt': 0.05, 'skeptical_doubt': 0.05, 'skeptical_doubt_hindrance': 0.05,
            'rites_rituals': 0.1, 'ignorance': 0.02,
            'form_desire': 0.03, 'formless_desire': 0.02,
            'craving': 0.01, 'clinging': 0.015
        }

        self.ripening_patterns = self._init_ripening_patterns()
        self.kilesa_interactions = self._init_kilesa_interactions()

        self.current_kilesas = KilesamState()
        self.accumulated_wholesome = 0.0
        self.accumulated_unwholesome = 0.0
        self.karmic_seeds: List[KarmicSeed] = []
        self.current_time = 0
        self.path_stage = PathStage.ORDINARY

        self.meditation_practices: List[MeditationPractice] = []
        self.current_meditation_suppression = 0.0
        self.current_wholesome_cultivation = 0.0

        self.rebirth_history: List[RebirthCircumstance] = []
        self.current_rebirth: Optional[RebirthCircumstance] = None

        self.current_context = {
            'stress_level': 0.5,
            'social_harmony': 0.5,
            'health_status': 0.5,
            'spiritual_environment': 0.5,
            'teaching_availability': 0.5
        }

        self.history_log: List[Dict] = []

    def _init_ripening_patterns(self) -> Dict[str, Dict[RipeningType, float]]:
        return {
            'hatred': {RipeningType.IMMEDIATE: 0.4, RipeningType.NEXT_LIFE: 0.5,
                      RipeningType.LATER_LIVES: 0.1, RipeningType.INDEFINITE: 0.0},
            'anger': {RipeningType.IMMEDIATE: 0.45, RipeningType.NEXT_LIFE: 0.45,
                     RipeningType.LATER_LIVES: 0.1, RipeningType.INDEFINITE: 0.0},
            'ill_will': {RipeningType.IMMEDIATE: 0.4, RipeningType.NEXT_LIFE: 0.5,
                        RipeningType.LATER_LIVES: 0.1, RipeningType.INDEFINITE: 0.0},
            'greed': {RipeningType.IMMEDIATE: 0.2, RipeningType.NEXT_LIFE: 0.6,
                     RipeningType.LATER_LIVES: 0.2, RipeningType.INDEFINITE: 0.0},
            'sensual_desire': {RipeningType.IMMEDIATE: 0.3, RipeningType.NEXT_LIFE: 0.6,
                              RipeningType.LATER_LIVES: 0.1, RipeningType.INDEFINITE: 0.0},
            'envy': {RipeningType.IMMEDIATE: 0.3, RipeningType.NEXT_LIFE: 0.6,
                    RipeningType.LATER_LIVES: 0.1, RipeningType.INDEFINITE: 0.0},
            'delusion': {RipeningType.IMMEDIATE: 0.05, RipeningType.NEXT_LIFE: 0.2,
                        RipeningType.LATER_LIVES: 0.75, RipeningType.INDEFINITE: 0.0},
            'ignorance': {RipeningType.IMMEDIATE: 0.03, RipeningType.NEXT_LIFE: 0.15,
                         RipeningType.LATER_LIVES: 0.82, RipeningType.INDEFINITE: 0.0},
            'conceit': {RipeningType.IMMEDIATE: 0.25, RipeningType.NEXT_LIFE: 0.45,
                       RipeningType.LATER_LIVES: 0.3, RipeningType.INDEFINITE: 0.0},
            'pride': {RipeningType.IMMEDIATE: 0.25, RipeningType.NEXT_LIFE: 0.45,
                     RipeningType.LATER_LIVES: 0.3, RipeningType.INDEFINITE: 0.0},
            'wrong_view': {RipeningType.IMMEDIATE: 0.1, RipeningType.NEXT_LIFE: 0.3,
                          RipeningType.LATER_LIVES: 0.6, RipeningType.INDEFINITE: 0.0},
            'doubt': {RipeningType.IMMEDIATE: 0.05, RipeningType.NEXT_LIFE: 0.25,
                     RipeningType.LATER_LIVES: 0.7, RipeningType.INDEFINITE: 0.0},
            'restlessness': {RipeningType.IMMEDIATE: 0.3, RipeningType.NEXT_LIFE: 0.4,
                            RipeningType.LATER_LIVES: 0.3, RipeningType.INDEFINITE: 0.0},
            'default': {RipeningType.IMMEDIATE: 0.2, RipeningType.NEXT_LIFE: 0.4,
                       RipeningType.LATER_LIVES: 0.4, RipeningType.INDEFINITE: 0.0}
        }

    def _init_kilesa_interactions(self) -> Dict[Tuple[str, str], float]:
        return {
            ('hatred', 'conceit'): 1.5,
            ('hatred', 'wrong_view'): 1.4,
            ('anger', 'hostility'): 1.6,
            ('ill_will', 'envy'): 1.4,
            ('greed', 'delusion'): 1.2,
            ('sensual_desire', 'covetousness'): 1.3,
            ('stinginess', 'envy'): 1.4,
            ('craving', 'clinging'): 1.8,
            ('conceit', 'arrogance'): 1.4,
            ('wrong_view', 'personality_view'): 1.6,
            ('doubt', 'restlessness'): 1.3,
            ('hypocrisy', 'fraud'): 1.5,
            ('shamelessness', 'recklessness'): 1.4,
            ('sloth_torpor', 'restlessness'): -0.7,
            ('mental_torpor', 'restlessness_worry'): -0.6,
            ('delusion', 'greed'): 1.1,
            ('delusion', 'hatred'): 1.1,
            ('ignorance', 'wrong_view'): 1.3,
        }

    def perform_action(self, intention_strength: float, active_kilesas: Dict[str, float],
                      object_weight: float = 1.0, wholesome: bool = False) -> List[KarmicSeed]:
        active_kilesas = self._apply_path_modifications(active_kilesas)
        effective_intensities = self._calculate_kilesa_interactions(active_kilesas)
        effective_intensities = self._apply_meditation_effects(effective_intensities)

        seeds = []
        for kilesa_type, intensity in effective_intensities.items():
            if intensity > 0:
                ripening_probs = self.ripening_patterns.get(kilesa_type,
                                                          self.ripening_patterns['default'])
                trigger_conditions = self._generate_trigger_conditions(kilesa_type,
                                                                      self.current_context)
                seed = KarmicSeed(
                    kilesa_type=kilesa_type,
                    intensity=intention_strength * intensity,
                    creation_time=self.current_time,
                    wholesome=wholesome,
                    ripening_probabilities=ripening_probs,
                    decay_rate=self.kilesa_decay_rates.get(kilesa_type, 0.1),
                    object_weight=object_weight,
                    trigger_conditions=trigger_conditions
                )
                seeds.append(seed)
                self.karmic_seeds.append(seed)

        self._update_kilesa_states(active_kilesas, intention_strength)

        total_unwholesome = sum(s.intensity * s.object_weight for s in seeds if not s.wholesome)
        total_wholesome = sum(s.intensity * s.object_weight for s in seeds if s.wholesome)
        total_wholesome += self.current_wholesome_cultivation * intention_strength

        if wholesome or total_wholesome > 0:
            self.accumulated_wholesome += total_wholesome
        else:
            self.accumulated_unwholesome += total_unwholesome

        self._log_action(seeds, intention_strength, active_kilesas, wholesome)
        return seeds

    def _apply_path_modifications(self, active_kilesas: Dict[str, float]) -> Dict[str, float]:
        modified = active_kilesas.copy()

        if self.path_stage == PathStage.STREAM_ENTRY:
            modified['personality_view'] = 0.0
            modified['doubt'] = 0.0
            modified['skeptical_doubt'] = 0.0
            modified['rites_rituals'] = 0.0
            for key in modified:
                if key not in ['personality_view', 'doubt', 'skeptical_doubt', 'rites_rituals']:
                    modified[key] *= 0.6

        elif self.path_stage == PathStage.ONCE_RETURNER:
            modified['personality_view'] = 0.0
            modified['doubt'] = 0.0
            modified['rites_rituals'] = 0.0
            modified['sensual_desire'] *= 0.3
            modified['ill_will'] *= 0.3

        elif self.path_stage == PathStage.NON_RETURNER:
            modified['personality_view'] = 0.0
            modified['doubt'] = 0.0
            modified['rites_rituals'] = 0.0
            modified['sensual_desire'] = 0.0
            modified['ill_will'] = 0.0
            modified['greed'] = 0.0
            modified['anger'] = 0.0
            modified['hatred'] = 0.0

        elif self.path_stage == PathStage.ARAHANT:
            modified = {key: 0.0 for key in modified}

        return modified

    def _calculate_kilesa_interactions(self, active_kilesas: Dict[str, float]) -> Dict[str, float]:
        effective = active_kilesas.copy()

        for (k1, k2), multiplier in self.kilesa_interactions.items():
            if k1 in effective and k2 in effective:
                if multiplier > 0:
                    if multiplier > 1:
                        combined_effect = effective[k1] * effective[k2] * (multiplier - 1)
                        effective[k1] += combined_effect * 0.5
                        effective[k2] += combined_effect * 0.5
                    else:
                        combined_effect = (effective[k1] + effective[k2]) * multiplier
                        effective[k1] = combined_effect * 0.5
                        effective[k2] = combined_effect * 0.5
                else:
                    suppression = abs(multiplier)
                    if effective[k1] > effective[k2]:
                        effective[k2] = max(0, effective[k2] - suppression * effective[k1])
                    else:
                        effective[k1] = max(0, effective[k1] - suppression * effective[k2])

        return effective

    def _apply_meditation_effects(self, kilesa_intensities: Dict[str, float]) -> Dict[str, float]:
        if not self.meditation_practices:
            return kilesa_intensities

        modified = kilesa_intensities.copy()

        for practice in self.meditation_practices:
            suppression = practice.get_kilesa_suppression_factor()

            if practice.practice_type == MeditationType.METTA:
                hatred_kilesas = ['hatred', 'anger', 'ill_will', 'hostility', 'aversion']
                for kilesa in hatred_kilesas:
                    if kilesa in modified:
                        modified[kilesa] *= (1 - suppression * 1.5)

            elif practice.practice_type == MeditationType.VIPASSANA:
                delusion_kilesas = ['delusion', 'ignorance', 'wrong_view', 'doubt', 'personality_view']
                for kilesa in delusion_kilesas:
                    if kilesa in modified:
                        modified[kilesa] *= (1 - suppression * 1.3)
                for kilesa in modified:
                    modified[kilesa] *= (1 - suppression * 0.8)

            elif practice.practice_type == MeditationType.SAMATHA:
                agitation_kilesas = ['restlessness', 'worry', 'mental_restlessness', 'restlessness_worry']
                for kilesa in agitation_kilesas:
                    if kilesa in modified:
                        modified[kilesa] *= (1 - suppression * 1.4)

        self.current_meditation_suppression = sum(p.get_kilesa_suppression_factor()
                                                 for p in self.meditation_practices) / max(len(self.meditation_practices), 1)
        self.current_wholesome_cultivation = sum(p.get_wholesome_cultivation_factor()
                                               for p in self.meditation_practices) / max(len(self.meditation_practices), 1)

        return {k: max(0, v) for k, v in modified.items()}

    def _generate_trigger_conditions(self, kilesa_type: str, context: Dict) -> RipeningTrigger:
        trigger_patterns = {
            'hatred': {
                'situational_factors': {'conflict_present': 0.8, 'injustice_perceived': 0.9},
                'mental_state_factors': {'stress_level': 0.7, 'tiredness': 0.6},
                'social_context': {'interpersonal_tension': 0.8}
            },
            'greed': {
                'situational_factors': {'wealth_disparity': 0.7, 'luxury_exposure': 0.8},
                'mental_state_factors': {'dissatisfaction': 0.9, 'insecurity': 0.6},
                'social_context': {'competitive_environment': 0.7}
            },
            'doubt': {
                'situational_factors': {'teaching_contradictions': 0.8, 'spiritual_confusion': 0.9},
                'mental_state_factors': {'uncertainty': 0.9, 'intellectual_pride': 0.5},
                'social_context': {'conflicting_opinions': 0.7}
            },
        }

        pattern = trigger_patterns.get(kilesa_type, {
            'situational_factors': {'general_stress': 0.5},
            'mental_state_factors': {'emotional_instability': 0.5},
            'social_context': {'social_pressure': 0.5}
        })

        return RipeningTrigger(
            situational_factors=pattern.get('situational_factors', {}),
            mental_state_factors=pattern.get('mental_state_factors', {}),
            karmic_readiness=random.uniform(0.3, 0.8),
            astrological_factors=random.uniform(0.8, 1.2),
            social_context=pattern.get('social_context', {}),
            health_factors={'physical_stress': context.get('stress_level', 0.5)}
        )

    def _log_action(self, seeds: List[KarmicSeed], intention_strength: float,
                   active_kilesas: Dict[str, float], wholesome: bool):
        self.history_log.append({
            'time': self.current_time,
            'action_type': 'wholesome' if wholesome else 'unwholesome',
            'intention_strength': intention_strength,
            'kilesas_activated': list(active_kilesas.keys()),
            'seeds_created': len(seeds),
            'total_accumulated_unwholesome': self.accumulated_unwholesome,
            'total_accumulated_wholesome': self.accumulated_wholesome,
            'meditation_suppression': self.current_meditation_suppression,
            'path_stage': self.path_stage.value,
            'active_kilesa_count': len([k for k, v in self.current_kilesas.to_dict().items() if v > 0.01])
        })

    def _update_kilesa_states(self, active_kilesas: Dict[str, float], intention_strength: float):
        current_dict = self.current_kilesas.to_dict()
        for kilesa, intensity in active_kilesas.items():
            if kilesa in current_dict:
                reinforcement = 0.1 * intention_strength * intensity
                current_dict[kilesa] = min(1.0, current_dict[kilesa] + reinforcement)
        self.current_kilesas.from_dict(current_dict)

    def advance_time(self, time_steps: int = 1, context_updates: Optional[Dict] = None):
        self.current_time += time_steps
        if context_updates:
            self.current_context.update(context_updates)
        ripening_events = self._check_sophisticated_ripening()
        self._log_state()
        return ripening_events

    def _check_sophisticated_ripening(self) -> List[Dict]:
        ripening_events = []
        for seed in self.karmic_seeds:
            if seed.current_strength(self.current_time) < 0.01:
                continue
            ripening_prob = seed.calculate_ripening_probability(
                {**self.current_context, 'time': self.current_time},
                self.current_meditation_suppression
            )
            if random.random() < ripening_prob:
                ripening_type = self._determine_ripening_type(seed)
                event = {
                    'time': self.current_time,
                    'seed': seed,
                    'ripening_type': ripening_type,
                    'intensity': seed.current_strength(self.current_time),
                    'observable_effect': self._generate_observable_effect(seed, ripening_type),
                    'context_factors': self.current_context.copy(),
                    'meditation_influence': self.current_meditation_suppression
                }
                seed.ripening_history.append({
                    'time': self.current_time,
                    'type': ripening_type,
                    'intensity': event['intensity'],
                    'context': self.current_context.copy()
                })
                ripening_events.append(event)
        return ripening_events

    def _log_state(self):
        active_seeds = [s for s in self.karmic_seeds if s.current_strength(self.current_time) > 0.01]
        self.history_log.append({
            'time': self.current_time,
            'action_type': 'state_update',
            'total_accumulated_unwholesome': self.accumulated_unwholesome,
            'total_accumulated_wholesome': self.accumulated_wholesome,
            'active_seeds_count': len(active_seeds),
            'meditation_suppression': self.current_meditation_suppression,
            'path_stage': self.path_stage.value,
            'strongest_kilesas': sorted([(k, v) for k, v in self.current_kilesas.to_dict().items() if v > 0.01],
                                       key=lambda x: x[1], reverse=True)[:5],
            'context': self.current_context.copy()
        })

    def _determine_ripening_type(self, seed: KarmicSeed) -> RipeningType:
        probs = seed.ripening_probabilities
        rand = random.random()
        cumulative = 0
        for ripening_type, prob in probs.items():
            cumulative += prob
            if rand <= cumulative:
                return ripening_type
        return RipeningType.INDEFINITE

    def _generate_observable_effect(self, seed: KarmicSeed, ripening_type: RipeningType) -> str:
        effects_map = {
            'hatred': {
                RipeningType.IMMEDIATE: "Conflict arises, anger directed at self",
                RipeningType.NEXT_LIFE: "Born in circumstances of hostility",
                RipeningType.LATER_LIVES: "Habitual anger patterns across lives"
            },
            'conceit': {
                RipeningType.IMMEDIATE: "Social humiliation or status loss",
                RipeningType.NEXT_LIFE: "Born in circumstances affecting self-image",
                RipeningType.LATER_LIVES: "Pride patterns creating repeated suffering"
            },
            'doubt': {
                RipeningType.IMMEDIATE: "Missed spiritual opportunities",
                RipeningType.NEXT_LIFE: "Born away from Dhamma teachings",
                RipeningType.LATER_LIVES: "Systematic unavailability of liberation paths"
            }
        }

        default_effect = f"Karmic ripening of {seed.kilesa_type} ({ripening_type.value})"
        return effects_map.get(seed.kilesa_type, {}).get(ripening_type, default_effect)

    def set_path_stage(self, stage: PathStage):
        self.path_stage = stage

        if stage != PathStage.ORDINARY:
            current_dict = self.current_kilesas.to_dict()

            if stage in [PathStage.STREAM_ENTRY, PathStage.ONCE_RETURNER,
                        PathStage.NON_RETURNER, PathStage.ARAHANT]:
                current_dict['personality_view'] = 0.0
                current_dict['doubt'] = 0.0
                current_dict['skeptical_doubt'] = 0.0
                current_dict['rites_rituals'] = 0.0

            if stage in [PathStage.NON_RETURNER, PathStage.ARAHANT]:
                current_dict['sensual_desire'] = 0.0
                current_dict['ill_will'] = 0.0
                current_dict['greed'] = 0.0
                current_dict['anger'] = 0.0
                current_dict['hatred'] = 0.0
                current_dict['envy'] = 0.0
                current_dict['stinginess'] = 0.0

            if stage == PathStage.ARAHANT:
                current_dict = {key: 0.0 for key in current_dict}

            self.current_kilesas.from_dict(current_dict)

    def add_meditation_practice(self, practice: MeditationPractice):
        self.meditation_practices.append(practice)

    def remove_meditation_practice(self, practice_type: MeditationType):
        self.meditation_practices = [p for p in self.meditation_practices
                                   if p.practice_type != practice_type]

    def calculate_rebirth_circumstances(self) -> RebirthCircumstance:
        unwholesome_strength = self.accumulated_unwholesome
        wholesome_strength = self.accumulated_wholesome

        active_kilesas = [(k, v) for k, v in self.current_kilesas.to_dict().items() if v > 0.1]
        active_kilesas.sort(key=lambda x: x[1], reverse=True)

        karmic_balance = wholesome_strength - unwholesome_strength

        if karmic_balance > 2.0:
            if self.path_stage in [PathStage.NON_RETURNER, PathStage.ARAHANT]:
                realm = RebirthRealm.BRAHMA
            else:
                realm = RebirthRealm.DEVA
        elif karmic_balance > 0:
            realm = RebirthRealm.HUMAN
        elif karmic_balance > -1.0:
            realm = RebirthRealm.GHOST
        elif karmic_balance > -2.0:
            realm = RebirthRealm.ANIMAL
        else:
            realm = RebirthRealm.HELL

        primary_kilesas = [k for k, v in active_kilesas[:3]]

        if 'conceit' in primary_kilesas or 'arrogance' in primary_kilesas:
            family_conditions = "Born into high status but with pride-related challenges"
        elif 'stinginess' in primary_kilesas or 'greed' in primary_kilesas:
            family_conditions = "Born into materially poor conditions"
        elif 'hatred' in primary_kilesas or 'anger' in primary_kilesas:
            family_conditions = "Born into conflict-prone or violent environment"
        else:
            family_conditions = "Average social and family conditions"

        if 'vanity' in primary_kilesas:
            physical_conditions = "Physical beauty but attachment to appearance"
        elif 'negligence' in primary_kilesas or 'sloth_torpor' in primary_kilesas:
            physical_conditions = "Health issues, chronic fatigue, or physical limitations"
        elif wholesome_strength > unwholesome_strength:
            physical_conditions = "Good health and physical capabilities"
        else:
            physical_conditions = "Average physical constitution"

        if 'wrong_view' in primary_kilesas or 'doubt' in primary_kilesas:
            mental_conditions = "Intellectual confusion, difficulty understanding truth"
        elif 'delusion' in primary_kilesas:
            mental_conditions = "Limited intellectual capacity, prone to misunderstanding"
        elif self.path_stage != PathStage.ORDINARY:
            mental_conditions = "High wisdom potential, clear understanding"
        else:
            mental_conditions = "Average intellectual and wisdom capabilities"

        if self.path_stage != PathStage.ORDINARY:
            spiritual_conditions = "Easy access to Dhamma teachings and qualified teachers"
        elif 'rites_rituals' in primary_kilesas:
            spiritual_conditions = "Access to spiritual teachings but tendency toward ritualism"
        elif 'doubt' in primary_kilesas:
            spiritual_conditions = "Exposure to many teachings but difficulty choosing path"
        else:
            spiritual_conditions = "Limited but sufficient access to spiritual guidance"

        environmental_conditions = f"Conditions suitable for {realm.value} existence"

        lifespan_ranges = {
            RebirthRealm.HELL: (500, 10000),
            RebirthRealm.ANIMAL: (1, 100),
            RebirthRealm.GHOST: (500, 5000),
            RebirthRealm.HUMAN: (20, 120),
            RebirthRealm.DEVA: (1000, 100000),
            RebirthRealm.BRAHMA: (10000, 1000000)
        }

        life_span_range = lifespan_ranges.get(realm, (50, 80))
        karmic_momentum = min(1.0, (abs(karmic_balance) + len(active_kilesas) * 0.1))

        return RebirthCircumstance(
            realm=realm,
            family_conditions=family_conditions,
            physical_conditions=physical_conditions,
            mental_conditions=mental_conditions,
            spiritual_conditions=spiritual_conditions,
            environmental_conditions=environmental_conditions,
            primary_kilesas=primary_kilesas,
            life_span_range=life_span_range,
            karmic_momentum=karmic_momentum
        )

    def simulate_rebirth(self) -> RebirthCircumstance:
        rebirth = self.calculate_rebirth_circumstances()
        self.rebirth_history.append(rebirth)
        self.current_rebirth = rebirth
        self.current_kilesas = KilesamState()

        for kilesa in rebirth.primary_kilesas:
            if hasattr(self.current_kilesas, kilesa):
                setattr(self.current_kilesas, kilesa, 0.3)

        return rebirth

    def visualize_karmic_evolution(self, save_path: Optional[str] = None):
        if not self.history_log:
            print("No history data available for visualization")
            return

        df = pd.DataFrame(self.history_log)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Karmic accumulation over time
        ax1 = axes[0, 0]
        state_data = df[df['action_type'] == 'state_update']
        ax1.plot(state_data['time'], state_data['total_accumulated_unwholesome'], 'r-', label='Unwholesome Karma', linewidth=2)
        ax1.plot(state_data['time'], state_data['total_accumulated_wholesome'], 'g-', label='Wholesome Karma', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Accumulated Karma')
        ax1.set_title('Karmic Accumulation Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Active seeds count over time
        ax2 = axes[0, 1]
        ax2.plot(state_data['time'], state_data['active_seeds_count'], 'b-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Active Karmic Seeds')
        ax2.set_title('Active Karmic Seeds Over Time')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Meditation effects over time
        ax3 = axes[1, 0]
        if 'meditation_suppression' in state_data.columns:
            ax3.plot(state_data['time'], state_data['meditation_suppression'], 'purple', linewidth=2)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Meditation Suppression Factor')
            ax3.set_title('Meditation Practice Effects')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)

        # Plot 4: Path progression
        ax4 = axes[1, 1]
        path_stages = ['ordinary', 'stream_entry', 'once_returner', 'non_returner', 'arahant']
        path_progression = []
        times = []

        for _, row in state_data.iterrows():
            times.append(row['time'])
            path_progression.append(path_stages.index(row['path_stage']))

        ax4.step(times, path_progression, where='post', linewidth=2, color='orange')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Path Stage')
        ax4.set_yticks(range(len(path_stages)))
        ax4.set_yticklabels([s.replace('_', ' ').title() for s in path_stages])
        ax4.set_title('Spiritual Path Progression')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

    def visualize_kilesa_patterns(self, save_path: Optional[str] = None):
        if not self.history_log:
            print("No history data available for visualization")
            return

        kilesa_timeline = []
        for entry in self.history_log:
            if entry['action_type'] not in ['wholesome', 'unwholesome']:
                continue
            time_point = entry['time']
            for kilesa in entry.get('kilesas_activated', []):
                kilesa_timeline.append({
                    'time': time_point,
                    'kilesa': kilesa,
                    'intensity': entry['intention_strength']
                })

        if not kilesa_timeline:
            print("No kilesa activation data available")
            return

        kilesa_df = pd.DataFrame(kilesa_timeline)
        pivot_table = kilesa_df.pivot_table(
            values='intensity',
            index='kilesa',
            columns='time',
            aggfunc='sum',
            fill_value=0
        )

        plt.figure(figsize=(15, 10))
        sns.heatmap(pivot_table, cmap='Reds', cbar_kws={'label': 'Activation Intensity'})
        plt.title('Kilesa Activation Patterns Over Time')
        plt.xlabel('Time')
        plt.ylabel('Kilesa Type')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Kilesa patterns visualization saved to {save_path}")
        else:
            plt.show()

    def generate_rebirth_report(self) -> str:
        if not self.current_rebirth:
            rebirth = self.calculate_rebirth_circumstances()
        else:
            rebirth = self.current_rebirth

        report = f"""
=== DETAILED REBIRTH CIRCUMSTANCES REPORT ===

REALM: {rebirth.realm.value.upper()}
Karmic Momentum: {rebirth.karmic_momentum:.2f}
Expected Lifespan: {rebirth.life_span_range[0]} - {rebirth.life_span_range[1]} years

FAMILY & SOCIAL CONDITIONS:
{rebirth.family_conditions}

PHYSICAL CONDITIONS:
{rebirth.physical_conditions}

MENTAL CONDITIONS:
{rebirth.mental_conditions}

SPIRITUAL CONDITIONS:
{rebirth.spiritual_conditions}

ENVIRONMENTAL CONDITIONS:
{rebirth.environmental_conditions}

PRIMARY KILESA TENDENCIES:
{', '.join(rebirth.primary_kilesas) if rebirth.primary_kilesas else 'None dominant'}

CURRENT KARMIC STATE:
- Accumulated Wholesome: {self.accumulated_wholesome:.2f}
- Accumulated Unwholesome: {self.accumulated_unwholesome:.2f}
- Karmic Balance: {self.accumulated_wholesome - self.accumulated_unwholesome:.2f}
- Path Attainment: {self.path_stage.value.replace('_', ' ').title()}

MEDITATION PRACTICE INFLUENCE:
- Current Suppression Factor: {self.current_meditation_suppression:.2f}
- Wholesome Cultivation: {self.current_wholesome_cultivation:.2f}
- Active Practices: {len(self.meditation_practices)}

KARMIC SEEDS STATUS:
- Total Active Seeds: {len([s for s in self.karmic_seeds if s.current_strength(self.current_time) > 0.01])}
- Strongest Seeds: {', '.join([s.kilesa_type for s in sorted(self.karmic_seeds, key=lambda x: x.current_strength(self.current_time), reverse=True)[:5]])}

=== END REPORT ===
        """

        return report.strip()

    def get_state_summary(self) -> Dict:
        active_seeds = [s for s in self.karmic_seeds if s.current_strength(self.current_time) > 0.01]

        return {
            'time': self.current_time,
            'path_stage': self.path_stage.value,
            'accumulated_wholesome': self.accumulated_wholesome,
            'accumulated_unwholesome': self.accumulated_unwholesome,
            'active_kilesas': {k: v for k, v in self.current_kilesas.to_dict().items() if v > 0.01},
            'active_seeds_count': len(active_seeds),
            'strongest_seeds': sorted(active_seeds,
                                    key=lambda x: x.current_strength(self.current_time),
                                    reverse=True)[:5]
        }

# Example usage
if __name__ == "__main__":
    karma_model = TheravadaKarmaHMM()

    print("=== THERAVADA KARMA HMM SIMULATION ===\n")

    # Add meditation practices
    vipassana = MeditationPractice(MeditationType.VIPASSANA, 1.5, 0.8, 0.6, 2.0, 0.7, 200)
    metta = MeditationPractice(MeditationType.METTA, 0.5, 0.9, 0.7, 1.0, 0.6, 50)
    karma_model.add_meditation_practice(vipassana)
    karma_model.add_meditation_practice(metta)

    print(f"Added meditation practices")

    # Perform actions
    karma_model.current_context.update({'stress_level': 0.8, 'conflict_present': 0.9})

    seeds1 = karma_model.perform_action(0.9, {'hatred': 0.8, 'anger': 0.7, 'hostility': 0.5}, 1.0, False)
    print(f"Hatred action created {len(seeds1)} seeds")

    seeds2 = karma_model.perform_action(0.7, {'conceit': 0.8, 'arrogance': 0.6, 'vanity': 0.5}, 1.2, False)
    print(f"Pride action created {len(seeds2)} seeds")

    seeds3 = karma_model.perform_action(0.8, {}, 1.0, True)
    print(f"Wholesome action enhanced by meditation")

    # Time progression
    print(f"\nAdvancing time with dynamic context...")
    ripening_events = []
    for i in range(30):
        karma_model.current_context.update({
            'stress_level': 0.3 + 0.4 * np.sin(i * 0.2),
            'social_harmony': 0.6 + 0.3 * np.cos(i * 0.15)
        })
        events = karma_model.advance_time(1)
        ripening_events.extend(events)

        for event in events:
            if event['intensity'] > 0.1:
                print(f"Time {event['time']}: {event['observable_effect']}")

    print(f"Total ripening events: {len(ripening_events)}")

    # Path progression
    karma_model.set_path_stage(PathStage.STREAM_ENTRY)
    print(f"\nAttained Stream Entry")

    seeds4 = karma_model.perform_action(0.8, {'doubt': 0.7, 'personality_view': 0.6, 'conceit': 0.5}, 1.0, False)
    print(f"Action after Stream Entry: {len(seeds4)} seeds (reduced)")

    # Rebirth calculation
    print(f"\nRebirth circumstances:")
    rebirth_report = karma_model.generate_rebirth_report()
    print(rebirth_report)

    # Visualization
    try:
        karma_model.visualize_karmic_evolution()
        karma_model.visualize_kilesa_patterns()
        print("Visualizations created")
    except Exception as e:
        print(f"Visualization not available: {e}")

    # Final state
    final_state = karma_model.get_state_summary()
    print(f"\nFinal state:")
    print(f"Path stage: {final_state['path_stage']}")
    print(f"Karmic balance: {karma_model.accumulated_wholesome - karma_model.accumulated_unwholesome:.2f}")
    print(f"Active practices: {len(karma_model.meditation_practices)}")

    print(f"\n=== SIMULATION COMPLETE ===")
