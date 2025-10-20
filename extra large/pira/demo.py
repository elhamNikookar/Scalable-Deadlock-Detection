#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
فایل دمو برای نمایش نحوه استفاده از شبیه‌سازهای TPMC
"""

import time
from tpmc_dining_philosophers import TPMCSimulator
from tpmc_advanced_simulation import AdvancedTPMCSimulator

def demo_basic_tpmc():
    """دمو نسخه پایه TPMC"""
    print("=" * 60)
    print("🎯 دمو نسخه پایه TPMC")
    print("=" * 60)
    
    # ایجاد شبیه‌ساز
    simulator = TPMCSimulator(num_philosophers=50)  # تعداد کمتر برای دمو
    
    # اجرای شبیه‌سازی
    start_time = time.time()
    results = simulator.run_tpmc_simulation()
    end_time = time.time()
    
    # نمایش نتایج
    print(f"\n📊 نتایج نسخه پایه:")
    print(f"   ⏱️  زمان اجرا: {end_time - start_time:.2f} ثانیه")
    print(f"   🔄 تعداد تکرارها: {results['total_iterations']}")
    print(f"   🔍 حالت‌های اکتشاف شده: {results['states_explored']}")
    print(f"   ⚠️  Deadlock‌های یافت شده: {results['deadlocks_found']}")
    
    # نمایش بصری
    simulator.visualize_system_state("demo_basic_results.png")
    print("   📈 نمودار در demo_basic_results.png ذخیره شد")
    
    return simulator

def demo_advanced_tpmc():
    """دمو نسخه پیشرفته TPMC"""
    print("\n" + "=" * 60)
    print("🚀 دمو نسخه پیشرفته TPMC")
    print("=" * 60)
    
    # ایجاد شبیه‌ساز پیشرفته
    simulator = AdvancedTPMCSimulator(
        num_philosophers=50,  # تعداد کمتر برای دمو
        deadlock_probability=0.5  # احتمال بالاتر برای دمو
    )
    
    # اجرای شبیه‌سازی
    start_time = time.time()
    results = simulator.run_advanced_tpmc_simulation()
    end_time = time.time()
    
    # نمایش نتایج
    print(f"\n📊 نتایج نسخه پیشرفته:")
    print(f"   ⏱️  زمان اجرا: {end_time - start_time:.2f} ثانیه")
    print(f"   🔄 تعداد تکرارها: {results['total_iterations']}")
    print(f"   🔍 حالت‌های اکتشاف شده: {results['states_explored']}")
    print(f"   ⚠️  Deadlock‌های یافت شده: {results['deadlocks_found']}")
    
    if results['performance_metrics']:
        print(f"   📊 نمره پایداری: {results['performance_metrics']['stability_score']:.3f}")
        print(f"   📊 میانگین رقابت منابع: {results['performance_metrics']['avg_resource_contention']:.3f}")
    
    # نمایش جزئیات deadlock‌ها
    if results['deadlocks_found'] > 0:
        print(f"\n🔍 جزئیات Deadlock‌ها:")
        for i, deadlock in enumerate(results['deadlock_states'][:3]):  # نمایش 3 مورد اول
            print(f"   Deadlock {i+1}:")
            print(f"      تکرار: {deadlock['iteration']}")
            print(f"      نوع: {deadlock['deadlock_info']['type']}")
            print(f"      شدت: {deadlock['deadlock_info']['severity']:.3f}")
            print(f"      چرخه‌ها: {len(deadlock['deadlock_info']['cycles'])}")
            print(f"      زنجیره‌های انتظار: {len(deadlock['deadlock_info']['waiting_chains'])}")
    
    # نمایش بصری
    simulator.visualize_advanced_system("demo_advanced_results.png")
    print("   📈 نمودار پیشرفته در demo_advanced_results.png ذخیره شد")
    
    # صادر کردن نتایج
    simulator.export_advanced_results("demo_advanced_results.json")
    print("   💾 نتایج در demo_advanced_results.json ذخیره شد")
    
    return simulator

def demo_comparison():
    """دمو مقایسه دو نسخه"""
    print("\n" + "=" * 60)
    print("⚖️  مقایسه نسخه پایه و پیشرفته")
    print("=" * 60)
    
    # تست با تعداد فیلسوفان مختلف
    philosopher_counts = [20, 30, 50]
    
    print(f"{'تعداد فیلسوفان':<15} {'نسخه پایه (ثانیه)':<20} {'نسخه پیشرفته (ثانیه)':<25}")
    print("-" * 60)
    
    for count in philosopher_counts:
        # نسخه پایه
        basic_sim = TPMCSimulator(num_philosophers=count)
        start_time = time.time()
        basic_sim.run_tpmc_simulation()
        basic_time = time.time() - start_time
        
        # نسخه پیشرفته
        advanced_sim = AdvancedTPMCSimulator(
            num_philosophers=count,
            deadlock_probability=0.3
        )
        start_time = time.time()
        advanced_sim.run_advanced_tpmc_simulation()
        advanced_time = time.time() - start_time
        
        print(f"{count:<15} {basic_time:<20.2f} {advanced_time:<25.2f}")

def demo_custom_parameters():
    """دمو با پارامترهای سفارشی"""
    print("\n" + "=" * 60)
    print("⚙️  دمو با پارامترهای سفارشی")
    print("=" * 60)
    
    # تست با احتمال‌های مختلف deadlock
    probabilities = [0.1, 0.3, 0.5, 0.7]
    
    print(f"{'احتمال Deadlock':<20} {'Deadlock‌های یافت شده':<25} {'زمان اجرا (ثانیه)':<20}")
    print("-" * 65)
    
    for prob in probabilities:
        simulator = AdvancedTPMCSimulator(
            num_philosophers=30,
            deadlock_probability=prob
        )
        
        start_time = time.time()
        results = simulator.run_advanced_tpmc_simulation()
        execution_time = time.time() - start_time
        
        print(f"{prob:<20} {results['deadlocks_found']:<25} {execution_time:<20.2f}")

def main():
    """تابع اصلی دمو"""
    print("🎭 دمو شبیه‌سازهای TPMC")
    print("بر اساس تحقیقات آقای پیرا")
    print("=" * 60)
    
    try:
        # دمو نسخه پایه
        basic_simulator = demo_basic_tpmc()
        
        # دمو نسخه پیشرفته
        advanced_simulator = demo_advanced_tpmc()
        
        # دمو مقایسه
        demo_comparison()
        
        # دمو پارامترهای سفارشی
        demo_custom_parameters()
        
        print("\n" + "=" * 60)
        print("✅ تمام دموها با موفقیت اجرا شدند!")
        print("📁 فایل‌های خروجی:")
        print("   - demo_basic_results.png")
        print("   - demo_advanced_results.png")
        print("   - demo_advanced_results.json")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ خطا در اجرای دمو: {e}")
        print("لطفاً مطمئن شوید که تمام کتابخانه‌های مورد نیاز نصب شده‌اند:")
        print("pip install numpy networkx matplotlib")

if __name__ == "__main__":
    main()
