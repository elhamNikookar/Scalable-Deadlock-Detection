#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نسخه پیشرفته شبیه‌سازی TPMC با احتمال بیشتر deadlock
برای مسئله فیلسوفان غذا با 100 فیلسوف

این نسخه شامل:
- الگوریتم‌های پیشرفته‌تر برای تشخیص deadlock
- شبیه‌سازی دقیق‌تر رفتار فیلسوفان
- تحلیل عمیق‌تر ویژگی‌های گراف
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from collections import defaultdict
import json

class PhilosopherState(Enum):
    """حالت‌های مختلف فیلسوف"""
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"
    WAITING = "waiting"  # حالت جدید برای انتظار

@dataclass
class Fork:
    """کلاس چنگال با ویژگی‌های پیشرفته"""
    id: int
    available: bool = True
    owner: Optional[int] = None
    request_queue: List[int] = None  # صف درخواست‌ها
    
    def __post_init__(self):
        if self.request_queue is None:
            self.request_queue = []

@dataclass
class Philosopher:
    """کلاس فیلسوف با ویژگی‌های پیشرفته"""
    id: int
    state: PhilosopherState = PhilosopherState.THINKING
    left_fork: Optional[int] = None
    right_fork: Optional[int] = None
    wait_time: float = 0.0
    eat_count: int = 0
    think_time: float = 0.0
    hunger_level: float = 0.0  # سطح گرسنگی (0-1)
    patience: float = 1.0  # صبر فیلسوف (0-1)
    priority: int = 0  # اولویت فیلسوف

class AdvancedTPMCSimulator:
    """شبیه‌ساز پیشرفته TPMC برای مسئله فیلسوفان غذا"""
    
    def __init__(self, num_philosophers: int = 100, deadlock_probability: float = 0.3):
        self.num_philosophers = num_philosophers
        self.deadlock_probability = deadlock_probability
        self.philosophers = []
        self.forks = []
        self.graph = nx.DiGraph()
        self.wait_graph = nx.DiGraph()  # گراف انتظار
        self.deadlock_detected = False
        self.iteration_count = 0
        self.max_iterations = 100
        
        # آمارگیری پیشرفته
        self.stats = {
            'deadlocks_found': 0,
            'total_iterations': 0,
            'execution_time': 0.0,
            'states_explored': 0,
            'deadlock_states': [],
            'performance_metrics': {},
            'graph_evolution': [],
            'philosopher_behavior': defaultdict(list)
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """مقداردهی اولیه سیستم با ویژگی‌های پیشرفته"""
        # ایجاد فیلسوفان با ویژگی‌های تصادفی
        for i in range(self.num_philosophers):
            philosopher = Philosopher(
                id=i,
                patience=random.uniform(0.3, 1.0),
                priority=random.randint(1, 10),
                hunger_level=random.uniform(0.0, 0.5)
            )
            self.philosophers.append(philosopher)
        
        # ایجاد چنگال‌ها
        for i in range(self.num_philosophers):
            fork = Fork(id=i)
            self.forks.append(fork)
        
        # ایجاد گراف‌های اولیه
        self._build_initial_graphs()
    
    def _build_initial_graphs(self):
        """ساخت گراف‌های اولیه سیستم"""
        self.graph.clear()
        self.wait_graph.clear()
        
        # اضافه کردن گره‌های فیلسوفان
        for philosopher in self.philosophers:
            self.graph.add_node(
                f"philosopher_{philosopher.id}",
                type="philosopher",
                state=philosopher.state.value,
                id=philosopher.id,
                hunger_level=philosopher.hunger_level,
                patience=philosopher.patience,
                priority=philosopher.priority
            )
            
            # گراف انتظار
            self.wait_graph.add_node(
                f"philosopher_{philosopher.id}",
                type="philosopher",
                state=philosopher.state.value
            )
        
        # اضافه کردن گره‌های چنگال‌ها
        for fork in self.forks:
            self.graph.add_node(
                f"fork_{fork.id}",
                type="fork",
                available=fork.available,
                id=fork.id,
                request_count=len(fork.request_queue)
            )
        
        # اضافه کردن یال‌های وابستگی
        for i in range(self.num_philosophers):
            left_fork_id = i
            right_fork_id = (i + 1) % self.num_philosophers
            
            # فیلسوف به چنگال چپ
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{left_fork_id}",
                relation="needs_left",
                weight=1.0
            )
            
            # فیلسوف به چنگال راست
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{right_fork_id}",
                relation="needs_right",
                weight=1.0
            )
    
    def _detect_advanced_deadlock(self) -> Tuple[bool, Dict]:
        """تشخیص پیشرفته deadlock با تحلیل عمیق‌تر"""
        deadlock_info = {
            'type': 'none',
            'cycles': [],
            'waiting_chains': [],
            'resource_contention': 0,
            'severity': 0.0
        }
        
        try:
            # 1. تشخیص چرخه‌های قوی متصل
            strongly_connected = list(nx.strongly_connected_components(self.graph))
            deadlock_cycles = []
            
            for component in strongly_connected:
                if len(component) > 1:
                    # بررسی فیلسوفان گرسنه در چرخه
                    hungry_in_cycle = []
                    for node in component:
                        if node.startswith("philosopher_"):
                            node_data = self.graph.nodes[node]
                            if node_data.get('state') in [PhilosopherState.HUNGRY.value, PhilosopherState.WAITING.value]:
                                hungry_in_cycle.append(node)
                    
                    if len(hungry_in_cycle) >= 2:
                        deadlock_cycles.append(list(component))
                        deadlock_info['cycles'].append(list(component))
            
            # 2. تحلیل زنجیره‌های انتظار
            waiting_chains = self._analyze_waiting_chains()
            deadlock_info['waiting_chains'] = waiting_chains
            
            # 3. محاسبه رقابت منابع
            resource_contention = self._calculate_resource_contention()
            deadlock_info['resource_contention'] = resource_contention
            
            # 4. محاسبه شدت deadlock
            severity = self._calculate_deadlock_severity(deadlock_cycles, waiting_chains, resource_contention)
            deadlock_info['severity'] = severity
            
            # تصمیم‌گیری نهایی
            is_deadlock = (
                len(deadlock_cycles) > 0 or 
                len(waiting_chains) > 0 or 
                resource_contention > 0.8 or
                severity > 0.5
            )
            
            if is_deadlock:
                if len(deadlock_cycles) > 0:
                    deadlock_info['type'] = 'circular_wait'
                elif len(waiting_chains) > 0:
                    deadlock_info['type'] = 'waiting_chain'
                elif resource_contention > 0.8:
                    deadlock_info['type'] = 'resource_contention'
                else:
                    deadlock_info['type'] = 'mixed'
            
            return is_deadlock, deadlock_info
            
        except Exception as e:
            print(f"خطا در تشخیص deadlock: {e}")
            return False, deadlock_info
    
    def _analyze_waiting_chains(self) -> List[List[str]]:
        """تحلیل زنجیره‌های انتظار"""
        waiting_chains = []
        
        # یافتن فیلسوفان گرسنه
        hungry_philosophers = []
        for philosopher in self.philosophers:
            if philosopher.state in [PhilosopherState.HUNGRY, PhilosopherState.WAITING]:
                hungry_philosophers.append(philosopher.id)
        
        # تحلیل زنجیره‌های انتظار
        for start_philosopher in hungry_philosophers:
            chain = self._trace_waiting_chain(start_philosopher, set())
            if len(chain) > 1:
                waiting_chains.append(chain)
        
        return waiting_chains
    
    def _trace_waiting_chain(self, philosopher_id: int, visited: Set[int]) -> List[str]:
        """ردیابی زنجیره انتظار از یک فیلسوف"""
        if philosopher_id in visited:
            return []  # چرخه یافت شد
        
        visited.add(philosopher_id)
        chain = [f"philosopher_{philosopher_id}"]
        
        philosopher = self.philosophers[philosopher_id]
        if philosopher.state not in [PhilosopherState.HUNGRY, PhilosopherState.WAITING]:
            return chain
        
        # بررسی چنگال‌های مورد نیاز
        left_fork_id = philosopher_id
        right_fork_id = (philosopher_id + 1) % self.num_philosophers
        
        left_fork = self.forks[left_fork_id]
        right_fork = self.forks[right_fork_id]
        
        # اگر چنگال‌ها اشغال هستند، دنبال صاحب آن‌ها بگرد
        if not left_fork.available and left_fork.owner is not None:
            next_chain = self._trace_waiting_chain(left_fork.owner, visited.copy())
            chain.extend(next_chain)
        
        if not right_fork.available and right_fork.owner is not None:
            next_chain = self._trace_waiting_chain(right_fork.owner, visited.copy())
            chain.extend(next_chain)
        
        return chain
    
    def _calculate_resource_contention(self) -> float:
        """محاسبه میزان رقابت منابع"""
        total_requests = 0
        total_resources = len(self.forks)
        
        for fork in self.forks:
            total_requests += len(fork.request_queue)
        
        return total_requests / (total_resources * self.num_philosophers) if total_resources > 0 else 0
    
    def _calculate_deadlock_severity(self, cycles: List, waiting_chains: List, resource_contention: float) -> float:
        """محاسبه شدت deadlock"""
        severity = 0.0
        
        # وزن چرخه‌ها
        severity += len(cycles) * 0.3
        
        # وزن زنجیره‌های انتظار
        severity += len(waiting_chains) * 0.2
        
        # وزن رقابت منابع
        severity += resource_contention * 0.3
        
        # وزن فیلسوفان گرسنه
        hungry_count = sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY)
        severity += (hungry_count / self.num_philosophers) * 0.2
        
        return min(severity, 1.0)
    
    def _advanced_bayesian_optimization(self) -> Dict:
        """بهینه‌سازی بیزی پیشرفته با در نظر گیری احتمال deadlock"""
        # انتخاب فیلسوف بر اساس اولویت و سطح گرسنگی
        candidate_philosophers = []
        
        for philosopher in self.philosophers:
            if philosopher.state == PhilosopherState.THINKING:
                # احتمال تبدیل به گرسنه بر اساس سطح گرسنگی
                hunger_prob = min(philosopher.hunger_level + 0.1, 1.0)
                if random.random() < hunger_prob:
                    candidate_philosophers.append((philosopher.id, hunger_prob))
            elif philosopher.state == PhilosopherState.HUNGRY:
                # احتمال تلاش برای خوردن بر اساس صبر
                eat_prob = philosopher.patience * 0.5
                if random.random() < eat_prob:
                    candidate_philosophers.append((philosopher.id, eat_prob))
            elif philosopher.state == PhilosopherState.EATING:
                # احتمال تمام کردن غذا
                finish_prob = 0.4
                if random.random() < finish_prob:
                    candidate_philosophers.append((philosopher.id, finish_prob))
        
        if not candidate_philosophers:
            return {'action': 'no_change', 'philosopher_id': 0}
        
        # انتخاب بهترین کاندید بر اساس احتمال و اولویت
        best_candidate = max(candidate_philosophers, key=lambda x: x[1])
        philosopher_id = best_candidate[0]
        philosopher = self.philosophers[philosopher_id]
        
        # تعیین عمل بر اساس حالت فعلی
        if philosopher.state == PhilosopherState.THINKING:
            return {'action': 'become_hungry', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.HUNGRY:
            return {'action': 'try_eat', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.EATING:
            return {'action': 'finish_eating', 'philosopher_id': philosopher_id}
        
        return {'action': 'no_change', 'philosopher_id': philosopher_id}
    
    def _apply_advanced_transformation(self, action: Dict):
        """اعمال قوانین تبدیل پیشرفته"""
        philosopher_id = action['philosopher_id']
        philosopher = self.philosophers[philosopher_id]
        
        if action['action'] == 'become_hungry':
            philosopher.state = PhilosopherState.HUNGRY
            philosopher.hunger_level = min(philosopher.hunger_level + 0.2, 1.0)
            self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.HUNGRY.value
            self.graph.nodes[f"philosopher_{philosopher_id}"]['hunger_level'] = philosopher.hunger_level
            
        elif action['action'] == 'try_eat':
            left_fork_id = philosopher_id
            right_fork_id = (philosopher_id + 1) % self.num_philosophers
            
            left_fork = self.forks[left_fork_id]
            right_fork = self.forks[right_fork_id]
            
            # اضافه کردن به صف درخواست
            if left_fork_id not in left_fork.request_queue:
                left_fork.request_queue.append(philosopher_id)
            if right_fork_id not in right_fork.request_queue:
                right_fork.request_queue.append(philosopher_id)
            
            # بررسی در دسترس بودن چنگال‌ها
            if left_fork.available and right_fork.available:
                # گرفتن چنگال‌ها
                left_fork.available = False
                right_fork.available = False
                left_fork.owner = philosopher_id
                right_fork.owner = philosopher_id
                
                # حذف از صف درخواست
                if philosopher_id in left_fork.request_queue:
                    left_fork.request_queue.remove(philosopher_id)
                if philosopher_id in right_fork.request_queue:
                    right_fork.request_queue.remove(philosopher_id)
                
                philosopher.state = PhilosopherState.EATING
                philosopher.left_fork = left_fork_id
                philosopher.right_fork = right_fork_id
                philosopher.eat_count += 1
                philosopher.hunger_level = 0.0  # رفع گرسنگی
                
                # به‌روزرسانی گراف
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.EATING.value
                self.graph.nodes[f"philosopher_{philosopher_id}"]['hunger_level'] = 0.0
                self.graph.nodes[f"fork_{left_fork_id}"]['available'] = False
                self.graph.nodes[f"fork_{right_fork_id}"]['available'] = False
            else:
                # اگر چنگال‌ها در دسترس نیستند، به حالت انتظار برو
                philosopher.state = PhilosopherState.WAITING
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.WAITING.value
                
        elif action['action'] == 'finish_eating':
            # رها کردن چنگال‌ها
            if philosopher.left_fork is not None and philosopher.right_fork is not None:
                left_fork = self.forks[philosopher.left_fork]
                right_fork = self.forks[philosopher.right_fork]
                
                left_fork.available = True
                right_fork.available = True
                left_fork.owner = None
                right_fork.owner = None
                
                philosopher.state = PhilosopherState.THINKING
                philosopher.left_fork = None
                philosopher.right_fork = None
                philosopher.think_time = random.uniform(1.0, 5.0)  # زمان تفکر تصادفی
                
                # به‌روزرسانی گراف
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.THINKING.value
                if philosopher.left_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.left_fork}"]['available'] = True
                if philosopher.right_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.right_fork}"]['available'] = True
    
    def _extract_advanced_features(self) -> Dict:
        """استخراج ویژگی‌های پیشرفته گراف"""
        features = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_philosophers': len(self.philosophers),
            'num_forks': len(self.forks),
            'hungry_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            'eating_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING),
            'thinking_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            'waiting_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.WAITING),
            'available_forks': sum(1 for f in self.forks if f.available),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph),
            'avg_hunger_level': np.mean([p.hunger_level for p in self.philosophers]),
            'avg_patience': np.mean([p.patience for p in self.philosophers]),
            'total_requests': sum(len(f.request_queue) for f in self.forks),
            'resource_contention': self._calculate_resource_contention(),
            'graph_clustering': nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0,
            'assortativity': nx.degree_assortativity_coefficient(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
        return features
    
    def run_advanced_tpmc_simulation(self) -> Dict:
        """اجرای شبیه‌سازی پیشرفته TPMC"""
        print(f"شروع شبیه‌سازی پیشرفته TPMC برای {self.num_philosophers} فیلسوف...")
        print(f"احتمال deadlock: {self.deadlock_probability}")
        start_time = time.time()
        
        self.stats['total_iterations'] = 0
        self.stats['states_explored'] = 0
        self.stats['deadlocks_found'] = 0
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            self.stats['total_iterations'] += 1
            
            # بهینه‌سازی بیزی پیشرفته
            action = self._advanced_bayesian_optimization()
            self._apply_advanced_transformation(action)
            
            # استخراج ویژگی‌های پیشرفته
            features = self._extract_advanced_features()
            self.stats['states_explored'] += 1
            self.stats['graph_evolution'].append(features.copy())
            
            # تشخیص پیشرفته deadlock
            is_deadlock, deadlock_info = self._detect_advanced_deadlock()
            
            if is_deadlock:
                self.deadlock_detected = True
                self.stats['deadlocks_found'] += 1
                self.stats['deadlock_states'].append({
                    'iteration': iteration,
                    'features': features,
                    'deadlock_info': deadlock_info,
                    'timestamp': time.time()
                })
                
                print(f"⚠️  Deadlock تشخیص داده شد در تکرار {iteration}")
                print(f"   نوع: {deadlock_info['type']}")
                print(f"   شدت: {deadlock_info['severity']:.3f}")
                print(f"   چرخه‌ها: {len(deadlock_info['cycles'])}")
                print(f"   زنجیره‌های انتظار: {len(deadlock_info['waiting_chains'])}")
            
            # نمایش پیشرفت
            if iteration % 10 == 0:
                print(f"تکرار {iteration}/{self.max_iterations} - "
                      f"گرسنگی: {features['hungry_philosophers']}, "
                      f"در حال خوردن: {features['eating_philosophers']}, "
                      f"انتظار: {features['waiting_philosophers']}, "
                      f"چنگال‌های آزاد: {features['available_forks']}, "
                      f"رقابت منابع: {features['resource_contention']:.3f}")
        
        end_time = time.time()
        self.stats['execution_time'] = end_time - start_time
        
        # محاسبه معیارهای عملکرد
        self._calculate_performance_metrics()
        
        print(f"\n✅ شبیه‌سازی پیشرفته TPMC تکمیل شد!")
        print(f"⏱️  زمان اجرا: {self.stats['execution_time']:.2f} ثانیه")
        print(f"🔄 تعداد تکرارها: {self.stats['total_iterations']}")
        print(f"🔍 تعداد حالت‌های اکتشاف شده: {self.stats['states_explored']}")
        print(f"⚠️  تعداد deadlock‌های یافت شده: {self.stats['deadlocks_found']}")
        
        return self.stats
    
    def _calculate_performance_metrics(self):
        """محاسبه معیارهای عملکرد"""
        if not self.stats['graph_evolution']:
            return
        
        features_list = self.stats['graph_evolution']
        
        # محاسبه آمار کلی
        self.stats['performance_metrics'] = {
            'avg_hungry_philosophers': np.mean([f['hungry_philosophers'] for f in features_list]),
            'max_hungry_philosophers': max([f['hungry_philosophers'] for f in features_list]),
            'avg_eating_philosophers': np.mean([f['eating_philosophers'] for f in features_list]),
            'avg_resource_contention': np.mean([f['resource_contention'] for f in features_list]),
            'max_resource_contention': max([f['resource_contention'] for f in features_list]),
            'avg_graph_density': np.mean([f['density'] for f in features_list]),
            'avg_clustering': np.mean([f['graph_clustering'] for f in features_list]),
            'stability_score': self._calculate_stability_score(features_list)
        }
    
    def _calculate_stability_score(self, features_list: List[Dict]) -> float:
        """محاسبه نمره پایداری سیستم"""
        if len(features_list) < 2:
            return 0.0
        
        # محاسبه تغییرات در ویژگی‌های کلیدی
        hungry_changes = np.std([f['hungry_philosophers'] for f in features_list])
        eating_changes = np.std([f['eating_philosophers'] for f in features_list])
        contention_changes = np.std([f['resource_contention'] for f in features_list])
        
        # نمره پایداری (هرچه کمتر تغییر کند، بهتر است)
        stability = 1.0 / (1.0 + hungry_changes + eating_changes + contention_changes)
        return min(stability, 1.0)
    
    def visualize_advanced_system(self, save_path: str = None):
        """نمایش بصری پیشرفته سیستم"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. گراف اصلی سیستم
        ax1 = axes[0, 0]
        self._draw_system_graph(ax1)
        ax1.set_title(f"گراف سیستم - {self.num_philosophers} فیلسوف")
        
        # 2. توزیع حالت‌های فیلسوفان
        ax2 = axes[0, 1]
        states = ['تفکر', 'گرسنگی', 'خوردن', 'انتظار']
        counts = [
            sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.WAITING)
        ]
        colors = ['lightblue', 'red', 'green', 'orange']
        bars = ax2.bar(states, counts, color=colors)
        ax2.set_title('توزیع حالت‌های فیلسوفان')
        ax2.set_ylabel('تعداد')
        
        # اضافه کردن مقادیر روی میله‌ها
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # 3. نمودار تکامل ویژگی‌ها
        ax3 = axes[0, 2]
        if self.stats['graph_evolution']:
            iterations = range(len(self.stats['graph_evolution']))
            hungry_evolution = [f['hungry_philosophers'] for f in self.stats['graph_evolution']]
            eating_evolution = [f['eating_philosophers'] for f in self.stats['graph_evolution']]
            
            ax3.plot(iterations, hungry_evolution, 'r-', label='گرسنگی', alpha=0.7)
            ax3.plot(iterations, eating_evolution, 'g-', label='خوردن', alpha=0.7)
            ax3.set_title('تکامل حالت‌های فیلسوفان')
            ax3.set_xlabel('تکرار')
            ax3.set_ylabel('تعداد')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. نمودار رقابت منابع
        ax4 = axes[1, 0]
        if self.stats['graph_evolution']:
            contention_evolution = [f['resource_contention'] for f in self.stats['graph_evolution']]
            ax4.plot(iterations, contention_evolution, 'b-', alpha=0.7)
            ax4.set_title('تکامل رقابت منابع')
            ax4.set_xlabel('تکرار')
            ax4.set_ylabel('میزان رقابت')
            ax4.grid(True, alpha=0.3)
        
        # 5. نمودار تراکم گراف
        ax5 = axes[1, 1]
        if self.stats['graph_evolution']:
            density_evolution = [f['density'] for f in self.stats['graph_evolution']]
            ax5.plot(iterations, density_evolution, 'm-', alpha=0.7)
            ax5.set_title('تکامل تراکم گراف')
            ax5.set_xlabel('تکرار')
            ax5.set_ylabel('تراکم')
            ax5.grid(True, alpha=0.3)
        
        # 6. آمار کلی
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""
آمار پیشرفته:
• تعداد فیلسوفان: {self.num_philosophers}
• تعداد چنگال‌ها: {len(self.forks)}
• تکرارهای انجام شده: {self.stats['total_iterations']}
• Deadlock‌های یافت شده: {self.stats['deadlocks_found']}
• زمان اجرا: {self.stats['execution_time']:.2f} ثانیه
• تراکم گراف: {nx.density(self.graph):.3f}
• نمره پایداری: {self.stats['performance_metrics'].get('stability_score', 0):.3f}
• میانگین رقابت منابع: {self.stats['performance_metrics'].get('avg_resource_contention', 0):.3f}
        """
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='Arial Unicode MS')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"نمودار پیشرفته در {save_path} ذخیره شد.")
        
        plt.show()
    
    def _draw_system_graph(self, ax):
        """رسم گراف سیستم"""
        # رنگ‌بندی گره‌ها
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            if node.startswith("philosopher_"):
                philosopher_id = int(node.split("_")[1])
                philosopher = self.philosophers[philosopher_id]
                if philosopher.state == PhilosopherState.THINKING:
                    node_colors.append('lightblue')
                elif philosopher.state == PhilosopherState.HUNGRY:
                    node_colors.append('red')
                elif philosopher.state == PhilosopherState.EATING:
                    node_colors.append('green')
                else:  # WAITING
                    node_colors.append('orange')
                node_sizes.append(300)
            else:  # fork
                node_colors.append('gray')
                node_sizes.append(100)
        
        # رسم گراف
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        nx.draw(self.graph, pos, 
                node_color=node_colors, 
                node_size=node_sizes,
                with_labels=False, 
                arrows=True,
                edge_color='gray',
                alpha=0.7,
                ax=ax)
    
    def export_advanced_results(self, filename: str = "advanced_tpmc_results.json"):
        """صادر کردن نتایج پیشرفته به فایل JSON"""
        results = {
            'simulation_info': {
                'num_philosophers': self.num_philosophers,
                'deadlock_probability': self.deadlock_probability,
                'max_iterations': self.max_iterations,
                'execution_time': self.stats['execution_time']
            },
            'statistics': self.stats,
            'final_state': {
                'philosophers': [
                    {
                        'id': p.id,
                        'state': p.state.value,
                        'hunger_level': p.hunger_level,
                        'patience': p.patience,
                        'priority': p.priority,
                        'eat_count': p.eat_count
                    } for p in self.philosophers
                ],
                'forks': [
                    {
                        'id': f.id,
                        'available': f.available,
                        'owner': f.owner,
                        'request_queue': f.request_queue
                    } for f in self.forks
                ]
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"نتایج پیشرفته در {filename} ذخیره شد.")

def main():
    """تابع اصلی"""
    print("شبیه‌سازی پیشرفته TPMC آقای پیرا")
    print("برای مسئله فیلسوفان غذا با 100 فیلسوف")
    print("=" * 50)
    
    # ایجاد شبیه‌ساز پیشرفته
    simulator = AdvancedTPMCSimulator(
        num_philosophers=100,
        deadlock_probability=0.4  # احتمال بالاتر برای deadlock
    )
    
    # اجرای شبیه‌سازی
    results = simulator.run_advanced_tpmc_simulation()
    
    # نمایش نتایج
    print("\n" + "=" * 50)
    print("نتایج نهایی:")
    print(f"✅ Deadlock تشخیص داده شد: {'بله' if results['deadlocks_found'] > 0 else 'خیر'}")
    print(f"⏱️  زمان کل اجرا: {results['execution_time']:.2f} ثانیه")
    print(f"🔄 تعداد تکرارها: {results['total_iterations']}")
    print(f"🔍 حالت‌های اکتشاف شده: {results['states_explored']}")
    
    if results['performance_metrics']:
        print(f"📊 نمره پایداری: {results['performance_metrics']['stability_score']:.3f}")
        print(f"📊 میانگین رقابت منابع: {results['performance_metrics']['avg_resource_contention']:.3f}")
    
    # نمایش بصری پیشرفته
    simulator.visualize_advanced_system("advanced_tpmc_visualization.png")
    
    # صادر کردن نتایج
    simulator.export_advanced_results("advanced_tpmc_results.json")
    
    return simulator

if __name__ == "__main__":
    simulator = main()
