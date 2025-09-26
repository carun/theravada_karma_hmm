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
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class TimeUnit(Enum):
    DAYS = "days"
    MONTHS = "months"
    YEARS = "years"
    LIFETIMES = "lifetimes"
    REBIRTHS = "rebirths"

class TimeScale:
    """Manages time scaling and conversion between different temporal units"""

    def __init__(self, base_unit: TimeUnit = TimeUnit.DAYS, scale_factor: float = 1.0):
        self.base_unit = base_unit
        self.scale_factor = scale_factor

        # Conversion factors to days
        self.conversion_to_days = {
            TimeUnit.DAYS: 1,
            TimeUnit.MONTHS: 30,
            TimeUnit.YEARS: 365,
            TimeUnit.LIFETIMES: 25550,  # ~70 years average
            TimeUnit.REBIRTHS: 25550   # Same as lifetimes for practical purposes
        }

    def convert_to_display_units(self, time_steps: int) -> float:
        """Convert internal time steps to display units"""
        return (time_steps * self.scale_factor) / self.conversion_to_days[self.base_unit]

    def convert_from_display_units(self, display_time: float) -> int:
        """Convert display units to internal time steps"""
        return int((display_time * self.conversion_to_days[self.base_unit]) / self.scale_factor)

    def get_display_label(self) -> str:
        """Get the appropriate label for time axis"""
        unit_labels = {
            TimeUnit.DAYS: "Days",
            TimeUnit.MONTHS: "Months",
            TimeUnit.YEARS: "Years",
            TimeUnit.LIFETIMES: "Lifetimes",
            TimeUnit.REBIRTHS: "Rebirths"
        }
        return unit_labels[self.base_unit]

    def format_time_point(self, time_steps: int) -> str:
        """Format a time point for display"""
        display_time = self.convert_to_display_units(time_steps)
        return f"{display_time:.2f} {self.base_unit.value}"

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
    def __init__(self, time_unit: TimeUnit = TimeUnit.DAYS, time_scale_factor: float = 1.0):
        self.time_scale = TimeScale(time_unit, time_scale_factor)
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

        # Log initial state
        self._log_state()

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

        # Add meditation cultivation bonus
        total_wholesome += self.current_wholesome_cultivation * intention_strength

        if wholesome:
            # Explicit wholesome action - always add positive karma
            base_wholesome_value = intention_strength * 0.5  # Base value for virtuous intentions
            total_wholesome = max(total_wholesome, base_wholesome_value)
            self.accumulated_wholesome += total_wholesome
        else:
            # Unwholesome action - add to unwholesome karma
            self.accumulated_unwholesome += max(total_unwholesome, intention_strength * 0.3)

        # Any wholesome cultivation from meditation still counts
        if total_wholesome > 0 and not wholesome:
            self.accumulated_wholesome += total_wholesome

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
        active_seeds = [s for s in self.karmic_seeds if s.current_strength(self.current_time) > 0.01]
        self.history_log.append({
            'time': self.current_time,
            'action_type': 'wholesome' if wholesome else 'unwholesome',
            'intention_strength': intention_strength,
            'kilesas_activated': list(active_kilesas.keys()),
            'seeds_created': len(seeds),
            'total_accumulated_unwholesome': self.accumulated_unwholesome,
            'total_accumulated_wholesome': self.accumulated_wholesome,
            'active_seeds_count': len(active_seeds),
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

                # Convert ripening seed energy to karma accumulation
                karma_value = event['intensity'] * 0.8  # 80% of seed strength becomes karma
                if seed.wholesome:
                    self.accumulated_wholesome += karma_value
                else:
                    self.accumulated_unwholesome += karma_value

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

        # Check for path progression
        self._check_path_progression()

        # Calculate research metrics
        karmic_balance = self.accumulated_wholesome - self.accumulated_unwholesome
        meditation_effectiveness = self._calculate_meditation_effectiveness()
        kilesa_suppression_rate = self._calculate_kilesa_suppression_rate()

        self.history_log.append({
            'time': self.current_time,
            'action_type': 'state_update',
            'total_accumulated_unwholesome': self.accumulated_unwholesome,
            'total_accumulated_wholesome': self.accumulated_wholesome,
            'karmic_balance': karmic_balance,
            'active_seeds_count': len(active_seeds),
            'meditation_suppression': self.current_meditation_suppression,
            'meditation_effectiveness': meditation_effectiveness,
            'kilesa_suppression_rate': kilesa_suppression_rate,
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

    def _check_path_progression(self):
        """Check if practitioner should advance to next path stage"""
        karmic_balance = self.accumulated_wholesome - self.accumulated_unwholesome
        meditation_strength = sum(p.get_kilesa_suppression_factor() for p in self.meditation_practices)

        # Progressive thresholds for path advancement
        if self.path_stage == PathStage.ORDINARY and karmic_balance > 5.0 and meditation_strength > 1.5:
            self.set_path_stage(PathStage.STREAM_ENTRY)
        elif self.path_stage == PathStage.STREAM_ENTRY and karmic_balance > 15.0 and meditation_strength > 3.0:
            self.set_path_stage(PathStage.ONCE_RETURNER)
        elif self.path_stage == PathStage.ONCE_RETURNER and karmic_balance > 30.0 and meditation_strength > 5.0:
            self.set_path_stage(PathStage.NON_RETURNER)
        elif self.path_stage == PathStage.NON_RETURNER and karmic_balance > 50.0 and meditation_strength > 8.0:
            # Additional check for complete kilesa eradication
            max_kilesa = max(self.current_kilesas.to_dict().values())
            if max_kilesa < 0.1:
                self.set_path_stage(PathStage.ARAHANT)

    def _calculate_meditation_effectiveness(self) -> float:
        """Calculate how effective current meditation practices are"""
        if not self.meditation_practices:
            return 0.0

        total_effectiveness = 0.0
        for practice in self.meditation_practices:
            # Factor in duration, consistency, quality, and guidance
            base_effectiveness = practice.daily_duration * practice.consistency * practice.quality
            experience_bonus = min(practice.years_practiced * 0.1, 1.0)  # Cap at 1.0 bonus
            guidance_bonus = practice.teacher_guidance * 0.5
            retreat_bonus = min(practice.retreat_hours * 0.001, 0.5)  # Cap at 0.5 bonus

            effectiveness = base_effectiveness * (1 + experience_bonus + guidance_bonus + retreat_bonus)
            total_effectiveness += effectiveness

        return min(total_effectiveness / len(self.meditation_practices), 10.0)  # Normalize and cap

    def _calculate_kilesa_suppression_rate(self) -> float:
        """Calculate rate at which kilesas are being suppressed"""
        current_kilesas = self.current_kilesas.to_dict()
        total_kilesa_strength = sum(current_kilesas.values())

        if total_kilesa_strength == 0:
            return 1.0  # Complete suppression

        # Calculate suppression based on meditation and path stage
        meditation_suppression = self.current_meditation_suppression
        path_suppression = {
            PathStage.ORDINARY: 0.0,
            PathStage.STREAM_ENTRY: 0.2,
            PathStage.ONCE_RETURNER: 0.4,
            PathStage.NON_RETURNER: 0.7,
            PathStage.ARAHANT: 1.0
        }.get(self.path_stage, 0.0)

        combined_suppression = min(meditation_suppression + path_suppression, 1.0)
        return combined_suppression

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

    def visualize_karmic_evolution(self):
        if not self.history_log:
            print("No history data available for visualization")
            return

        df = pd.DataFrame(self.history_log)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Convert time values for display
        state_data = df[df['action_type'] == 'state_update'].copy()
        display_times = [self.time_scale.convert_to_display_units(t) for t in state_data['time']]
        time_label = self.time_scale.get_display_label()

        # Plot 1: Karmic accumulation over time
        ax1 = axes[0, 0]
        ax1.plot(display_times, state_data['total_accumulated_unwholesome'], 'r-', label='Unwholesome Karma', linewidth=2)
        ax1.plot(display_times, state_data['total_accumulated_wholesome'], 'g-', label='Wholesome Karma', linewidth=2)
        ax1.set_xlabel(f'Time ({time_label})')
        ax1.set_ylabel('Accumulated Karma')
        ax1.set_title('Karmic Accumulation Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Active seeds count over time
        ax2 = axes[0, 1]
        ax2.plot(display_times, state_data['active_seeds_count'], 'b-', linewidth=2)
        ax2.set_xlabel(f'Time ({time_label})')
        ax2.set_ylabel('Active Karmic Seeds')
        ax2.set_title('Active Karmic Seeds Over Time')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Meditation effects over time
        ax3 = axes[1, 0]
        if 'meditation_suppression' in state_data.columns:
            ax3.plot(display_times, state_data['meditation_suppression'], 'purple', linewidth=2)
            ax3.set_xlabel(f'Time ({time_label})')
            ax3.set_ylabel('Meditation Suppression Factor')
            ax3.set_title('Meditation Practice Effects')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)

        # Plot 4: Path progression
        ax4 = axes[1, 1]
        path_stages = ['ordinary', 'stream_entry', 'once_returner', 'non_returner', 'arahant']
        path_progression = []

        for _, row in state_data.iterrows():
            path_progression.append(path_stages.index(row['path_stage']))

        ax4.step(display_times, path_progression, where='post', linewidth=2, color='orange')
        ax4.set_xlabel(f'Time ({time_label})')
        ax4.set_ylabel('Path Stage')
        ax4.set_yticks(range(len(path_stages)))
        ax4.set_yticklabels([s.replace('_', ' ').title() for s in path_stages])
        ax4.set_title('Spiritual Path Progression')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.show()

    def visualize_karmic_evolution_interactive(self, save_path: Optional[str] = None):
        """Create interactive plotly visualization of karmic evolution"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Using matplotlib fallback.")
            self.visualize_karmic_evolution()
            return

        if not self.history_log:
            print("No history data available for visualization")
            return

        df = pd.DataFrame(self.history_log)
        state_data = df[df['action_type'] == 'state_update'].copy()

        # Convert time values for display
        display_times = [self.time_scale.convert_to_display_units(t) for t in state_data['time']]
        time_label = self.time_scale.get_display_label()
        state_data['display_time'] = display_times

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Karmic Accumulation Over Time',
                'Active Karmic Seeds Over Time',
                'Meditation Practice Effects',
                'Spiritual Path Progression'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Plot 1: Karmic accumulation
        fig.add_trace(
            go.Scatter(
                x=state_data['display_time'],
                y=state_data['total_accumulated_unwholesome'],
                mode='lines',
                name='Unwholesome Karma',
                line=dict(color='red', width=2),
                hovertemplate=f'Time: %{{x:.2f}} {time_label}<br>Unwholesome: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=state_data['display_time'],
                y=state_data['total_accumulated_wholesome'],
                mode='lines',
                name='Wholesome Karma',
                line=dict(color='green', width=2),
                hovertemplate=f'Time: %{{x:.2f}} {time_label}<br>Wholesome: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Active seeds
        fig.add_trace(
            go.Scatter(
                x=state_data['display_time'],
                y=state_data['active_seeds_count'],
                mode='lines',
                name='Active Seeds',
                line=dict(color='blue', width=2),
                hovertemplate=f'Time: %{{x:.2f}} {time_label}<br>Active Seeds: %{{y}}<extra></extra>'
            ),
            row=1, col=2
        )

        # Plot 3: Meditation effects
        if 'meditation_suppression' in state_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=state_data['display_time'],
                    y=state_data['meditation_suppression'],
                    mode='lines',
                    name='Meditation Suppression',
                    line=dict(color='purple', width=2),
                    hovertemplate=f'Time: %{{x:.2f}} {time_label}<br>Suppression: %{{y:.3f}}<extra></extra>'
                ),
                row=2, col=1
            )

        # Plot 4: Path progression
        path_stages = ['ordinary', 'stream_entry', 'once_returner', 'non_returner', 'arahant']
        path_progression = [path_stages.index(row['path_stage']) for _, row in state_data.iterrows()]
        path_names = [row['path_stage'].replace('_', ' ').title() for _, row in state_data.iterrows()]

        fig.add_trace(
            go.Scatter(
                x=state_data['display_time'],
                y=path_progression,
                mode='lines+markers',
                name='Path Stage',
                line=dict(color='orange', width=2),
                customdata=path_names,
                hovertemplate=f'Time: %{{x:.2f}} {time_label}<br>Stage: %{{customdata}}<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Interactive Karmic Evolution Dashboard",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        # Update axes labels
        fig.update_xaxes(title_text=f"Time ({time_label})", row=1, col=1)
        fig.update_xaxes(title_text=f"Time ({time_label})", row=1, col=2)
        fig.update_xaxes(title_text=f"Time ({time_label})", row=2, col=1)
        fig.update_xaxes(title_text=f"Time ({time_label})", row=2, col=2)

        fig.update_yaxes(title_text="Accumulated Karma", row=1, col=1)
        fig.update_yaxes(title_text="Active Seeds Count", row=1, col=2)
        fig.update_yaxes(title_text="Suppression Factor", row=2, col=1)
        fig.update_yaxes(
            title_text="Path Stage",
            tickmode='array',
            tickvals=list(range(len(path_stages))),
            ticktext=[s.replace('_', ' ').title() for s in path_stages],
            row=2, col=2
        )

        if save_path:
            html_path = f"{save_path}.html"
            fig.write_html(html_path)
            print(f"Interactive visualization saved to {html_path}")
        else:
            fig.show()

    def visualize_kilesa_patterns(self):
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

        # Convert time columns to display units
        time_label = self.time_scale.get_display_label()
        display_columns = {}
        for col in pivot_table.columns:
            display_time = self.time_scale.convert_to_display_units(col)
            display_columns[col] = f"{display_time:.1f}"
        pivot_table = pivot_table.rename(columns=display_columns)

        plt.figure(figsize=(15, 10))
        sns.heatmap(pivot_table, cmap='Reds', cbar_kws={'label': 'Activation Intensity'})
        plt.title('Kilesa Activation Patterns Over Time')
        plt.xlabel(f'Time ({time_label})')
        plt.ylabel('Kilesa Type')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.show()

    def visualize_kilesa_patterns_interactive(self, save_path: Optional[str] = None):
        """Create interactive plotly heatmap of kilesa patterns"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Using matplotlib fallback.")
            self.visualize_kilesa_patterns()
            return

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
                    'display_time': self.time_scale.convert_to_display_units(time_point),
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
            columns='display_time',
            aggfunc='sum',
            fill_value=0
        )

        time_label = self.time_scale.get_display_label()

        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[f"{t:.1f}" for t in pivot_table.columns],
            y=pivot_table.index,
            colorscale='Reds',
            colorbar=dict(title="Activation Intensity"),
            hovertemplate=f'Time: %{{x}} {time_label}<br>Kilesa: %{{y}}<br>Intensity: %{{z:.3f}}<extra></extra>'
        ))

        fig.update_layout(
            title="Interactive Kilesa Activation Patterns Over Time",
            xaxis_title=f"Time ({time_label})",
            yaxis_title="Kilesa Type",
            height=600,
            width=1000
        )

        if save_path:
            html_path = f"{save_path}_interactive.html"
            fig.write_html(html_path)
            print(f"Interactive kilesa patterns saved to {html_path}")
        else:
            fig.show()

    def visualize_kilesa_interaction_network(self, save_path: Optional[str] = None, layout_type: str = 'spring'):
        """Create network visualization of kilesa interactions"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Network visualization requires plotly.")
            return

        # Extract nodes and edges from kilesa interactions
        nodes = set()
        edges = []
        edge_weights = []
        edge_colors = []

        # Add all kilesas as nodes
        for kilesa in self.current_kilesas.to_dict().keys():
            nodes.add(kilesa)

        # Add interaction edges
        for (k1, k2), weight in self.kilesa_interactions.items():
            if k1 in nodes and k2 in nodes:
                edges.append((k1, k2))
                edge_weights.append(abs(weight))
                edge_colors.append('red' if weight > 0 else 'blue')
                nodes.add(k1)
                nodes.add(k2)

        # Create node positions using simple circular layout for now
        import math
        node_list = list(nodes)
        n_nodes = len(node_list)
        node_positions = {}

        if layout_type == 'circular':
            for i, node in enumerate(node_list):
                angle = 2 * math.pi * i / n_nodes
                node_positions[node] = (math.cos(angle), math.sin(angle))
        else:  # spring layout approximation
            # Simple grid layout as approximation
            grid_size = int(math.ceil(math.sqrt(n_nodes)))
            for i, node in enumerate(node_list):
                x = (i % grid_size) - grid_size/2
                y = (i // grid_size) - grid_size/2
                node_positions[node] = (x, y)

        # Create edge traces
        edge_traces = []
        for i, (k1, k2) in enumerate(edges):
            x0, y0 = node_positions[k1]
            x1, y1 = node_positions[k2]
            weight = edge_weights[i]
            color = edge_colors[i]

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=max(1, weight * 5),
                    color=color
                ),
                hoverinfo='text',
                text=f'{k1} ↔ {k2}<br>Weight: {self.kilesa_interactions.get((k1, k2), 0):.2f}',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []

        # Get current kilesa strengths for node coloring
        current_kilesas = self.current_kilesas.to_dict()

        for node in node_list:
            x, y = node_positions[node]
            node_x.append(x)
            node_y.append(y)
            intensity = current_kilesas.get(node, 0)
            node_colors.append(intensity)
            node_text.append(f'{node}<br>Current: {intensity:.3f}')

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[node.replace('_', '<br>') for node in node_list],
            textposition='middle center',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=20,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(
                    title="Current Kilesa Strength",
                    thickness=15,
                    len=0.7
                ),
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=8)
        )

        # Create figure
        fig = go.Figure()

        # Add edge traces
        for trace in edge_traces:
            fig.add_trace(trace)

        # Add node trace
        fig.add_trace(node_trace)

        # Add legend for edge colors
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', width=3),
            name='Reinforcing (+)',
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Suppressing (-)',
            showlegend=True
        ))

        fig.update_layout(
            title="Interactive Kilesa Interaction Network",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node size reflects current activation level<br>"
                         "Edge width reflects interaction strength<br>"
                         "Red = Reinforcing, Blue = Suppressing",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            width=1000
        )

        if save_path:
            html_path = f"{save_path}_network.html"
            fig.write_html(html_path)
            print(f"Interactive network visualization saved to {html_path}")
        else:
            fig.show()

    def get_visualization_json(self, viz_type: str = "evolution") -> dict:
        """Get visualization data in JSON format for web UI"""
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly not available"}

        if not self.history_log:
            return {"error": "No history data available"}

        if viz_type == "evolution":
            return self._get_evolution_json()
        elif viz_type == "kilesa_patterns":
            return self._get_kilesa_patterns_json()
        elif viz_type == "network":
            return self._get_network_json()
        else:
            return {"error": f"Unknown visualization type: {viz_type}"}

    def _get_evolution_json(self) -> dict:
        """Get karmic evolution data as JSON"""
        try:
            import pandas as pd
            df = pd.DataFrame(self.history_log)
            state_data = df[df['action_type'] == 'state_update'].copy()

            if state_data.empty:
                return {"error": "No state data available"}

            display_times = [self.time_scale.convert_to_display_units(t) for t in state_data['time']]
            time_label = self.time_scale.get_display_label()
            state_data['display_time'] = display_times

            # Create the same structure as the plotly visualization but return data
            path_stages = ['ordinary', 'stream_entry', 'once_returner', 'non_returner', 'arahant']
            path_progression = [path_stages.index(row['path_stage']) for _, row in state_data.iterrows()]

            return {
                "time": display_times,
                "time_label": time_label,
                "wholesome_karma": state_data['total_accumulated_wholesome'].tolist(),
                "unwholesome_karma": state_data['total_accumulated_unwholesome'].tolist(),
                "active_seeds": state_data['active_seeds_count'].tolist(),
                "meditation_suppression": state_data.get('meditation_suppression', [0]*len(display_times)),
                "path_progression": path_progression,
                "path_stage_labels": [s.replace('_', ' ').title() for s in path_stages]
            }
        except Exception as e:
            return {"error": f"Failed to generate evolution data: {str(e)}"}

    def _get_kilesa_patterns_json(self) -> dict:
        """Get kilesa patterns data as JSON"""
        try:
            import pandas as pd
            kilesa_timeline = []
            for entry in self.history_log:
                if entry['action_type'] not in ['wholesome', 'unwholesome']:
                    continue
                time_point = entry['time']
                display_time = self.time_scale.convert_to_display_units(time_point)
                for kilesa in entry.get('kilesas_activated', []):
                    kilesa_timeline.append({
                        'display_time': display_time,
                        'kilesa': kilesa,
                        'intensity': entry['intention_strength']
                    })

            if not kilesa_timeline:
                return {"error": "No kilesa activation data available"}

            kilesa_df = pd.DataFrame(kilesa_timeline)
            pivot_table = kilesa_df.pivot_table(
                values='intensity',
                index='kilesa',
                columns='display_time',
                aggfunc='sum',
                fill_value=0
            )

            return {
                "time_label": self.time_scale.get_display_label(),
                "kilesas": pivot_table.index.tolist(),
                "times": [f"{t:.1f}" for t in pivot_table.columns],
                "intensities": pivot_table.values.tolist()
            }
        except Exception as e:
            return {"error": f"Failed to generate kilesa patterns: {str(e)}"}

    def _get_network_json(self) -> dict:
        """Get network data as JSON"""
        try:
            nodes = []
            edges = []

            # Get current kilesa strengths for node coloring
            current_kilesas = self.current_kilesas.to_dict()

            # Add nodes for all kilesas that have some activation
            for kilesa, strength in current_kilesas.items():
                if strength > 0.001:  # Show even weak activations
                    nodes.append({
                        "id": kilesa,
                        "label": kilesa.replace('_', ' ').title(),
                        "value": strength,
                        "title": f"{kilesa}: {strength:.3f}"
                    })

            # Add edges for kilesa interactions
            for (k1, k2), weight in self.kilesa_interactions.items():
                if k1 in current_kilesas and k2 in current_kilesas:
                    if current_kilesas[k1] > 0.001 and current_kilesas[k2] > 0.001:
                        edges.append({
                            "from": k1,
                            "to": k2,
                            "value": abs(weight) * 5,  # Scale for visibility
                            "color": "#e74c3c" if weight > 0 else "#3498db",
                            "title": f"{k1} ↔ {k2}: {weight:.3f}",
                            "weight": weight
                        })

            return {
                "nodes": nodes,
                "edges": edges,
                "interaction_count": len(edges)
            }
        except Exception as e:
            return {"error": f"Failed to generate network data: {str(e)}"}

    def compare_scenarios(self, scenarios: List[Dict], time_steps: int = 30) -> Dict:
        """Compare different scenarios and return comparative analysis"""
        results = {}

        for scenario_name, config in scenarios:
            # Create a new instance for this scenario
            scenario_model = TheravadaKarmaHMM(
                time_unit=config.get('time_unit', TimeUnit.DAYS),
                time_scale_factor=config.get('time_scale_factor', 1.0)
            )

            # Set up meditation practices
            for practice_config in config.get('meditation_practices', []):
                practice = MeditationPractice(**practice_config)
                scenario_model.add_meditation_practice(practice)

            # Set path stage if specified
            if 'path_stage' in config:
                scenario_model.set_path_stage(config['path_stage'])

            # Perform actions as specified
            for action in config.get('actions', []):
                scenario_model.perform_action(
                    action['intention_strength'],
                    action['active_kilesas'],
                    action.get('object_weight', 1.0),
                    action.get('wholesome', False)
                )

            # Advance time with context updates
            for i in range(time_steps):
                context_updates = config.get('context_pattern', {})
                # Apply dynamic context if function provided
                if callable(context_updates):
                    context_updates = context_updates(i)

                scenario_model.advance_time(1, context_updates)

            # Store results
            final_state = scenario_model.get_state_summary()
            results[scenario_name] = {
                'model': scenario_model,
                'final_state': final_state,
                'accumulated_wholesome': scenario_model.accumulated_wholesome,
                'accumulated_unwholesome': scenario_model.accumulated_unwholesome,
                'karmic_balance': scenario_model.accumulated_wholesome - scenario_model.accumulated_unwholesome,
                'path_stage': scenario_model.path_stage.value,
                'active_kilesas': {k: v for k, v in scenario_model.current_kilesas.to_dict().items() if v > 0.01}
            }

        return results

    def visualize_scenario_comparison(self, comparison_results: Dict, save_path: Optional[str] = None):
        """Create comparative visualization of different scenarios"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Using simple comparison print.")
            self._print_scenario_comparison(comparison_results)
            return

        scenarios = list(comparison_results.keys())
        n_scenarios = len(scenarios)

        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Karmic Balance Comparison',
                'Path Stage Comparison',
                'Active Kilesas Count',
                'Karmic Evolution Timeline'
            )
        )

        # Plot 1: Karmic Balance
        wholesome_values = [comparison_results[s]['accumulated_wholesome'] for s in scenarios]
        unwholesome_values = [comparison_results[s]['accumulated_unwholesome'] for s in scenarios]
        balance_values = [comparison_results[s]['karmic_balance'] for s in scenarios]

        fig.add_trace(
            go.Bar(name='Wholesome', x=scenarios, y=wholesome_values, marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Unwholesome', x=scenarios, y=unwholesome_values, marker_color='red'),
            row=1, col=1
        )

        # Plot 2: Path Stages
        path_stages = ['ordinary', 'stream_entry', 'once_returner', 'non_returner', 'arahant']
        path_values = [path_stages.index(comparison_results[s]['path_stage']) for s in scenarios]

        fig.add_trace(
            go.Bar(x=scenarios, y=path_values, marker_color='orange', name='Path Stage'),
            row=1, col=2
        )

        # Plot 3: Active Kilesas Count
        kilesa_counts = [len(comparison_results[s]['active_kilesas']) for s in scenarios]

        fig.add_trace(
            go.Bar(x=scenarios, y=kilesa_counts, marker_color='purple', name='Active Kilesas'),
            row=2, col=1
        )

        # Plot 4: Timeline comparison (first scenario for now)
        if scenarios:
            first_scenario = scenarios[0]
            model = comparison_results[first_scenario]['model']
            df = pd.DataFrame(model.history_log)
            state_data = df[df['action_type'] == 'state_update']

            if not state_data.empty:
                display_times = [model.time_scale.convert_to_display_units(t) for t in state_data['time']]
                time_label = model.time_scale.get_display_label()

                fig.add_trace(
                    go.Scatter(
                        x=display_times,
                        y=state_data['total_accumulated_wholesome'],
                        mode='lines',
                        name=f'{first_scenario} - Wholesome',
                        line=dict(color='green')
                    ),
                    row=2, col=2
                )

                fig.add_trace(
                    go.Scatter(
                        x=display_times,
                        y=state_data['total_accumulated_unwholesome'],
                        mode='lines',
                        name=f'{first_scenario} - Unwholesome',
                        line=dict(color='red')
                    ),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title_text="Scenario Comparison Dashboard",
            height=800,
            showlegend=True
        )

        # Update y-axis for path stages
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(path_stages))),
            ticktext=[s.replace('_', ' ').title() for s in path_stages],
            row=1, col=2
        )

        if save_path:
            html_path = f"{save_path}_comparison.html"
            fig.write_html(html_path)
            print(f"Scenario comparison saved to {html_path}")
        else:
            fig.show()

    def _print_scenario_comparison(self, comparison_results: Dict):
        """Simple text-based scenario comparison"""
        print("\n=== SCENARIO COMPARISON RESULTS ===")
        for scenario_name, results in comparison_results.items():
            print(f"\n{scenario_name.upper()}:")
            print(f"  Karmic Balance: {results['karmic_balance']:.3f}")
            print(f"  Path Stage: {results['path_stage'].replace('_', ' ').title()}")
            print(f"  Active Kilesas: {len(results['active_kilesas'])}")
            print(f"  Strongest Kilesas: {list(results['active_kilesas'].keys())[:3]}")

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
    # Initialize with meaningful time units - simulating daily practice over months
    karma_model = TheravadaKarmaHMM(time_unit=TimeUnit.MONTHS, time_scale_factor=1.0)

    print("=== THERAVADA KARMA HMM SIMULATION ===")
    print(f"Time scale: {karma_model.time_scale.get_display_label()}\n")

    # Add meditation practices
    vipassana = MeditationPractice(MeditationType.VIPASSANA, 1.5, 0.8, 0.6, 2.0, 0.7, 200)
    metta = MeditationPractice(MeditationType.METTA, 0.5, 0.9, 0.7, 1.0, 0.6, 50)
    karma_model.add_meditation_practice(vipassana)
    karma_model.add_meditation_practice(metta)

    print(f"Added meditation practices")
    print(f"Current time: {karma_model.time_scale.format_time_point(karma_model.current_time)}")

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
        if PLOTLY_AVAILABLE:
            print("\nCreating interactive visualizations...")
            karma_model.visualize_karmic_evolution_interactive("karma_evolution_demo")
            karma_model.visualize_kilesa_patterns_interactive("kilesa_patterns_demo")
            karma_model.visualize_kilesa_interaction_network("network_demo")
            print("Interactive visualizations created")
        else:
            print("\nCreating basic visualizations...")
            karma_model.visualize_karmic_evolution()
            karma_model.visualize_kilesa_patterns()
            print("Basic visualizations displayed")
    except Exception as e:
        print(f"Visualization not available: {e}")

    # Final state
    final_state = karma_model.get_state_summary()
    print(f"\nFinal state:")
    print(f"Path stage: {final_state['path_stage']}")
    print(f"Karmic balance: {karma_model.accumulated_wholesome - karma_model.accumulated_unwholesome:.2f}")
    print(f"Active practices: {len(karma_model.meditation_practices)}")

    # Demonstrate scenario comparison
    print("\n=== SCENARIO COMPARISON DEMO ===")

    scenarios = [
        ("high_meditation", {
            'time_unit': TimeUnit.MONTHS,
            'meditation_practices': [
                {'practice_type': MeditationType.VIPASSANA, 'daily_duration': 2.0, 'consistency': 0.9,
                 'quality': 0.8, 'years_practiced': 5.0, 'teacher_guidance': 0.9, 'retreat_hours': 500},
                {'practice_type': MeditationType.METTA, 'daily_duration': 1.0, 'consistency': 0.9,
                 'quality': 0.8, 'years_practiced': 3.0, 'teacher_guidance': 0.8, 'retreat_hours': 200}
            ],
            'actions': [
                {'intention_strength': 0.8, 'active_kilesas': {}, 'wholesome': True}
            ],
            'context_pattern': {'stress_level': 0.3, 'spiritual_environment': 0.9}
        }),
        ("low_meditation", {
            'time_unit': TimeUnit.MONTHS,
            'meditation_practices': [
                {'practice_type': MeditationType.VIPASSANA, 'daily_duration': 0.5, 'consistency': 0.5,
                 'quality': 0.4, 'years_practiced': 1.0, 'teacher_guidance': 0.3, 'retreat_hours': 20}
            ],
            'actions': [
                {'intention_strength': 0.7, 'active_kilesas': {'anger': 0.6, 'greed': 0.5}, 'wholesome': False}
            ],
            'context_pattern': {'stress_level': 0.7, 'spiritual_environment': 0.3}
        }),
        ("no_meditation", {
            'time_unit': TimeUnit.MONTHS,
            'meditation_practices': [],
            'actions': [
                {'intention_strength': 0.8, 'active_kilesas': {'hatred': 0.8, 'conceit': 0.7}, 'wholesome': False},
                {'intention_strength': 0.6, 'active_kilesas': {'greed': 0.7, 'envy': 0.5}, 'wholesome': False}
            ],
            'context_pattern': {'stress_level': 0.8, 'spiritual_environment': 0.1}
        })
    ]

    try:
        comparison_results = karma_model.compare_scenarios(scenarios, time_steps=15)
        karma_model.visualize_scenario_comparison(comparison_results)
        print("Scenario comparison completed")
    except Exception as e:
        print(f"Scenario comparison failed: {e}")

    print(f"\n=== SIMULATION COMPLETE ===")
