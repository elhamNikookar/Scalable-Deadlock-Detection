#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPMC (Two-Phase Model Checking) Simulation by Mr. Pira
For Dining Philosophers Problem with 100 philosophers

Based on:
- Pira et al. (2017): GTS+BOA approach
- Pira et al. (2022): Two-phase model checking framework
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class PhilosopherState(Enum):
    """Different states of a philosopher"""
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"

@dataclass
class Fork:
    """Fork class"""
    id: int
    available: bool = True
    owner: int = None

@dataclass
class Philosopher:
    """Philosopher class"""
    id: int
    state: PhilosopherState = PhilosopherState.THINKING
    left_fork: int = None
    right_fork: int = None
    wait_time: float = 0.0
    eat_count: int = 0

class TPMCSimulator:
    """TPMC Simulator for Dining Philosophers Problem"""
    
    def __init__(self, num_philosophers: int = 100):
        self.num_philosophers = num_philosophers
        self.philosophers = []
        self.forks = []
        self.graph = nx.DiGraph()
        self.deadlock_detected = False
        self.iteration_count = 0
        self.max_iterations = 100  # طبق مقاله TPMC
        
        # آمارگیری
        self.stats = {
            'deadlocks_found': 0,
            'total_iterations': 0,
            'execution_time': 0.0,
            'states_explored': 0,
            'deadlock_states': []
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """مقداردهی اولیه سیستم"""
        # ایجاد فیلسوفان
        for i in range(self.num_philosophers):
            philosopher = Philosopher(id=i)
            self.philosophers.append(philosopher)
        
        # ایجاد چنگال‌ها
        for i in range(self.num_philosophers):
            fork = Fork(id=i)
            self.forks.append(fork)
        
        # ایجاد گراف اولیه
        self._build_initial_graph()
    
    def _build_initial_graph(self):
        """ساخت گراف اولیه سیستم"""
        self.graph.clear()
        
        # اضافه کردن گره‌های فیلسوفان
        for philosopher in self.philosophers:
            self.graph.add_node(
                f"philosopher_{philosopher.id}",
                type="philosopher",
                state=philosopher.state.value,
                id=philosopher.id
            )
        
        # اضافه کردن گره‌های چنگال‌ها
        for fork in self.forks:
            self.graph.add_node(
                f"fork_{fork.id}",
                type="fork",
                available=fork.available,
                id=fork.id
            )
        
        # اضافه کردن یال‌های وابستگی
        for i in range(self.num_philosophers):
            left_fork_id = i
            right_fork_id = (i + 1) % self.num_philosophers
            
            # فیلسوف به چنگال چپ
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{left_fork_id}",
                relation="needs_left"
            )
            
            # فیلسوف به چنگال راست
            self.graph.add_edge(
                f"philosopher_{i}",
                f"fork_{right_fork_id}",
                relation="needs_right"
            )
    
    def _detect_deadlock_cycle(self) -> bool:
        """تشخیص چرخه deadlock در گراف"""
        try:
            # جستجوی چرخه‌های قوی متصل
            strongly_connected = list(nx.strongly_connected_components(self.graph))
            
            for component in strongly_connected:
                if len(component) > 1:
                    # بررسی اینکه آیا این چرخه شامل فیلسوفان گرسنه است
                    hungry_philosophers = []
                    for node in component:
                        if node.startswith("philosopher_"):
                            node_data = self.graph.nodes[node]
                            if node_data.get('state') == PhilosopherState.HUNGRY.value:
                                hungry_philosophers.append(node)
                    
                    if len(hungry_philosophers) >= 2:
                        return True
            
            return False
        except:
            return False
    
    def _bayesian_optimization_step(self) -> Dict:
        """مرحله بهینه‌سازی بیزی (فاز دوم TPMC)"""
        # شبیه‌سازی الگوریتم بهینه‌سازی بیزی
        # در واقعیت، این شامل نمونه‌گیری از توزیع posterior و بهینه‌سازی acquisition function است
        
        # انتخاب تصادفی فیلسوف برای تغییر حالت
        philosopher_id = random.randint(0, self.num_philosophers - 1)
        philosopher = self.philosophers[philosopher_id]
        
        # احتمال تغییر حالت بر اساس الگوریتم بیزی
        if philosopher.state == PhilosopherState.THINKING:
            # احتمال تبدیل به گرسنه
            if random.random() < 0.3:  # 30% احتمال
                return {'action': 'become_hungry', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.HUNGRY:
            # احتمال تلاش برای خوردن
            if random.random() < 0.5:  # 50% احتمال
                return {'action': 'try_eat', 'philosopher_id': philosopher_id}
        elif philosopher.state == PhilosopherState.EATING:
            # احتمال بازگشت به تفکر
            if random.random() < 0.4:  # 40% احتمال
                return {'action': 'finish_eating', 'philosopher_id': philosopher_id}
        
        return {'action': 'no_change', 'philosopher_id': philosopher_id}
    
    def _apply_transformation_rule(self, action: Dict):
        """اعمال قوانین تبدیل گراف (فاز اول TPMC)"""
        philosopher_id = action['philosopher_id']
        philosopher = self.philosophers[philosopher_id]
        
        if action['action'] == 'become_hungry':
            philosopher.state = PhilosopherState.HUNGRY
            # به‌روزرسانی گراف
            self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.HUNGRY.value
            
        elif action['action'] == 'try_eat':
            left_fork_id = philosopher_id
            right_fork_id = (philosopher_id + 1) % self.num_philosophers
            
            left_fork = self.forks[left_fork_id]
            right_fork = self.forks[right_fork_id]
            
            # بررسی در دسترس بودن چنگال‌ها
            if left_fork.available and right_fork.available:
                # گرفتن چنگال‌ها
                left_fork.available = False
                right_fork.available = False
                left_fork.owner = philosopher_id
                right_fork.owner = philosopher_id
                
                philosopher.state = PhilosopherState.EATING
                philosopher.left_fork = left_fork_id
                philosopher.right_fork = right_fork_id
                philosopher.eat_count += 1
                
                # به‌روزرسانی گراف
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.EATING.value
                self.graph.nodes[f"fork_{left_fork_id}"]['available'] = False
                self.graph.nodes[f"fork_{right_fork_id}"]['available'] = False
                
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
                
                # به‌روزرسانی گراف
                self.graph.nodes[f"philosopher_{philosopher_id}"]['state'] = PhilosopherState.THINKING.value
                if philosopher.left_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.left_fork}"]['available'] = True
                if philosopher.right_fork is not None:
                    self.graph.nodes[f"fork_{philosopher.right_fork}"]['available'] = True
    
    def _extract_graph_features(self) -> Dict:
        """استخراج ویژگی‌های گراف برای تحلیل"""
        features = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_philosophers': len(self.philosophers),
            'num_forks': len(self.forks),
            'hungry_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            'eating_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING),
            'thinking_philosophers': sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            'available_forks': sum(1 for f in self.forks if f.available),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph)
        }
        return features
    
    def run_tpmc_simulation(self) -> Dict:
        """اجرای شبیه‌سازی TPMC"""
        print(f"شروع شبیه‌سازی TPMC برای {self.num_philosophers} فیلسوف...")
        start_time = time.time()
        
        self.stats['total_iterations'] = 0
        self.stats['states_explored'] = 0
        self.stats['deadlocks_found'] = 0
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            self.stats['total_iterations'] += 1
            
            # فاز اول: اعمال قوانین تبدیل گراف
            action = self._bayesian_optimization_step()
            self._apply_transformation_rule(action)
            
            # استخراج ویژگی‌های گراف
            features = self._extract_graph_features()
            self.stats['states_explored'] += 1
            
            # تشخیص deadlock
            if self._detect_deadlock_cycle():
                self.deadlock_detected = True
                self.stats['deadlocks_found'] += 1
                self.stats['deadlock_states'].append({
                    'iteration': iteration,
                    'features': features,
                    'timestamp': time.time()
                })
                
                print(f"⚠️  Deadlock تشخیص داده شد در تکرار {iteration}")
                print(f"   ویژگی‌های گراف: {features}")
                
                # در صورت تشخیص deadlock، ادامه می‌دهیم تا الگوریتم کامل شود
                # (در واقعیت، ممکن است الگوریتم متوقف شود)
            
            # نمایش پیشرفت
            if iteration % 10 == 0:
                print(f"تکرار {iteration}/{self.max_iterations} - "
                      f"گرسنگی: {features['hungry_philosophers']}, "
                      f"در حال خوردن: {features['eating_philosophers']}, "
                      f"چنگال‌های آزاد: {features['available_forks']}")
        
        end_time = time.time()
        self.stats['execution_time'] = end_time - start_time
        
        print(f"\n✅ شبیه‌سازی TPMC تکمیل شد!")
        print(f"⏱️  زمان اجرا: {self.stats['execution_time']:.2f} ثانیه")
        print(f"🔄 تعداد تکرارها: {self.stats['total_iterations']}")
        print(f"🔍 تعداد حالت‌های اکتشاف شده: {self.stats['states_explored']}")
        print(f"⚠️  تعداد deadlock‌های یافت شده: {self.stats['deadlocks_found']}")
        
        return self.stats
    
    def visualize_system_state(self, save_path: str = None):
        """نمایش بصری وضعیت سیستم"""
        plt.figure(figsize=(15, 10))
        
        # ایجاد subplot برای گراف
        ax1 = plt.subplot(2, 2, 1)
        
        # رنگ‌بندی گره‌ها بر اساس نوع
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
                else:  # EATING
                    node_colors.append('green')
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
                ax=ax1)
        
        ax1.set_title(f"وضعیت سیستم - {self.num_philosophers} فیلسوف")
        
        # نمودار آمار
        ax2 = plt.subplot(2, 2, 2)
        states = ['تفکر', 'گرسنگی', 'خوردن']
        counts = [
            sum(1 for p in self.philosophers if p.state == PhilosopherState.THINKING),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.HUNGRY),
            sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING)
        ]
        ax2.bar(states, counts, color=['lightblue', 'red', 'green'])
        ax2.set_title('توزیع حالت‌های فیلسوفان')
        ax2.set_ylabel('تعداد')
        
        # نمودار چنگال‌ها
        ax3 = plt.subplot(2, 2, 3)
        fork_states = ['آزاد', 'اشغال شده']
        fork_counts = [
            sum(1 for f in self.forks if f.available),
            sum(1 for f in self.forks if not f.available)
        ]
        ax3.bar(fork_states, fork_counts, color=['lightgreen', 'orange'])
        ax3.set_title('وضعیت چنگال‌ها')
        ax3.set_ylabel('تعداد')
        
        # آمار کلی
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = f"""
آمار کلی:
• تعداد فیلسوفان: {self.num_philosophers}
• تعداد چنگال‌ها: {len(self.forks)}
• تکرارهای انجام شده: {self.stats['total_iterations']}
• Deadlock‌های یافت شده: {self.stats['deadlocks_found']}
• زمان اجرا: {self.stats['execution_time']:.2f} ثانیه
• تراکم گراف: {nx.density(self.graph):.3f}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='Arial Unicode MS')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"نمودار در {save_path} ذخیره شد.")
        
        plt.show()
    
    def export_results(self, filename: str = "tpmc_results.txt"):
        """صادر کردن نتایج به فایل"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("نتایج شبیه‌سازی TPMC برای مسئله فیلسوفان غذا\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"تعداد فیلسوفان: {self.num_philosophers}\n")
            f.write(f"تعداد چنگال‌ها: {len(self.forks)}\n")
            f.write(f"تعداد تکرارها: {self.stats['total_iterations']}\n")
            f.write(f"زمان اجرا: {self.stats['execution_time']:.2f} ثانیه\n")
            f.write(f"تعداد حالت‌های اکتشاف شده: {self.stats['states_explored']}\n")
            f.write(f"تعداد deadlock‌های یافت شده: {self.stats['deadlocks_found']}\n\n")
            
            f.write("جزئیات Deadlock‌ها:\n")
            f.write("-" * 30 + "\n")
            for i, deadlock in enumerate(self.stats['deadlock_states']):
                f.write(f"Deadlock {i+1}:\n")
                f.write(f"  تکرار: {deadlock['iteration']}\n")
                f.write(f"  ویژگی‌ها: {deadlock['features']}\n")
                f.write(f"  زمان: {deadlock['timestamp']}\n\n")
        
        print(f"نتایج در {filename} ذخیره شد.")

def main():
    """تابع اصلی"""
    print("شبیه‌سازی رویکرد TPMC آقای پیرا")
    print("برای مسئله فیلسوفان غذا با 100 فیلسوف")
    print("=" * 50)
    
    # ایجاد شبیه‌ساز
    simulator = TPMCSimulator(num_philosophers=100)
    
    # اجرای شبیه‌سازی
    results = simulator.run_tpmc_simulation()
    
    # نمایش نتایج
    print("\n" + "=" * 50)
    print("نتایج نهایی:")
    print(f"✅ Deadlock تشخیص داده شد: {'بله' if results['deadlocks_found'] > 0 else 'خیر'}")
    print(f"⏱️  زمان کل اجرا: {results['execution_time']:.2f} ثانیه")
    print(f"🔄 تعداد تکرارها: {results['total_iterations']}")
    print(f"🔍 حالت‌های اکتشاف شده: {results['states_explored']}")
    
    # نمایش بصری
    simulator.visualize_system_state("tpmc_visualization.png")
    
    # صادر کردن نتایج
    simulator.export_results("tpmc_results.txt")
    
    return simulator

if __name__ == "__main__":
    simulator = main()
