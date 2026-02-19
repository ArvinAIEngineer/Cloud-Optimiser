import streamlit as st
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import re

# AWS EC2 Pricing 
AWS_INSTANCES = {
    't3.micro':   {'vcpu': 2, 'ram': 1,  'cost': 0.0104},
    't3.small':   {'vcpu': 2, 'ram': 2,  'cost': 0.0208},
    't3.medium':  {'vcpu': 2, 'ram': 4,  'cost': 0.0416},
    't3.large':   {'vcpu': 2, 'ram': 8,  'cost': 0.0832},
    'c5.large':   {'vcpu': 2, 'ram': 4,  'cost': 0.0850},
    'm5.large':   {'vcpu': 2, 'ram': 8,  'cost': 0.0960},
    'c5.xlarge':  {'vcpu': 4, 'ram': 8,  'cost': 0.1700},
    'r5.large':   {'vcpu': 2, 'ram': 16, 'cost': 0.1260},
    'm5.xlarge':  {'vcpu': 4, 'ram': 16, 'cost': 0.1920},
    'm5.2xlarge': {'vcpu': 8, 'ram': 32, 'cost': 0.3840},
}

def parse_resource_string(resource_str):
    """Parse CPU/memory strings like '0.5', '1.0', '512M', '2G'"""
    if resource_str is None:
        return None
    
    resource_str = str(resource_str).strip()
    
    # CPU (just a number or with 'cpus')
    if 'cpus' not in resource_str.lower() and 'm' not in resource_str.lower() and 'g' not in resource_str.lower():
        try:
            return float(resource_str)
        except:
            return None
    
    # Memory
    if 'm' in resource_str.lower() or 'g' in resource_str.lower():
        # Extract number
        match = re.search(r'([\d.]+)', resource_str)
        if match:
            value = float(match.group(1))
            if 'g' in resource_str.lower():
                return value  # GB
            elif 'm' in resource_str.lower():
                return value / 1024  # MB to GB
    
    return None

def parse_docker_compose(compose_text: str) -> List[Dict]:
    """Parse docker-compose.yml and extract service requirements"""
    try:
        compose_data = yaml.safe_load(compose_text)
    except Exception as e:
        st.error(f"❌ Invalid YAML format: {e}")
        return []
    
    services = []
    
    if 'services' not in compose_data:
        st.error("❌ No 'services' section found in docker-compose.yml")
        return []
    
    for service_name, service_config in compose_data['services'].items():
        cpu = None
        memory = None
        
        # Try to extract from deploy.resources.limits
        if isinstance(service_config, dict):
            deploy = service_config.get('deploy', {})
            if isinstance(deploy, dict):
                resources = deploy.get('resources', {})
                if isinstance(resources, dict):
                    limits = resources.get('limits', {})
                    if isinstance(limits, dict):
                        cpu = parse_resource_string(limits.get('cpus'))
                        memory = parse_resource_string(limits.get('memory'))
        
        # Defaults if not specified
        if cpu is None:
            cpu = 0.5  # Default: 0.5 vCPU
        if memory is None:
            memory = 0.5  # Default: 512 MB
        
        services.append({
            'name': service_name,
            'cpu': cpu,
            'memory': memory
        })
    
    return services

def fitness_function(selected_instances: List[str], services: List[Dict]) -> float:
    """Calculate fitness (cost) for selected instance allocation"""
    total_cost = 0.0
    
    for service, instance_name in zip(services, selected_instances):
        if instance_name not in AWS_INSTANCES:
            return 999.0  # Invalid
        
        instance = AWS_INSTANCES[instance_name]
        
        # Check if instance can handle the service
        if service['cpu'] > instance['vcpu'] or service['memory'] > instance['ram']:
            return 999.0  # Instance too small
        
        total_cost += instance['cost']
    
    return total_cost

def binary_ga_optimize(services: List[Dict], max_iterations: int = 50, pop_size: int = 50):
    """
    Optimized Binary GA with YOUR research findings:
    - pC = 1.0 (always crossover)
    - μ = 0.02 (mutation rate)
    - Tournament selection
    - 50 iterations, 50 population
    """
    
    instance_names = list(AWS_INSTANCES.keys())
    n_services = len(services)
    
    # Initialize population (random instance assignment for each service)
    population = []
    for _ in range(pop_size):
        chromosome = [np.random.choice(instance_names) for _ in range(n_services)]
        population.append(chromosome)
    
    best_solution = None
    best_fitness = 999.0
    convergence = []
    
    for iteration in range(max_iterations):
        # Evaluate fitness
        fitness_scores = [fitness_function(chrom, services) for chrom in population]
        
        # Track best
        min_idx = np.argmin(fitness_scores)
        if fitness_scores[min_idx] < best_fitness:
            best_fitness = fitness_scores[min_idx]
            best_solution = population[min_idx].copy()
        
        convergence.append(best_fitness)
        
        # Selection (Tournament with k=3)
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            idx1, idx2, idx3 = np.random.choice(pop_size, 3, replace=False)
            tournament = [
                (population[idx1], fitness_scores[idx1]),
                (population[idx2], fitness_scores[idx2]),
                (population[idx3], fitness_scores[idx3])
            ]
            winner = min(tournament, key=lambda x: x[1])[0]
            new_population.append(winner.copy())
        
        # Crossover (pC = 1.0 - always crossover)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = new_population[i]
            parent2 = new_population[i+1] if i+1 < pop_size else new_population[0]
            
            # Three-point crossover (from your paper)
            if n_services >= 3:
                points = sorted(np.random.choice(range(1, n_services), 3, replace=False))
                child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:points[2]] + parent2[points[2]:]
                child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:points[2]] + parent1[points[2]:]
            else:
                # Uniform crossover for small service counts
                child1 = [parent1[j] if np.random.rand() < 0.5 else parent2[j] for j in range(n_services)]
                child2 = [parent2[j] if np.random.rand() < 0.5 else parent1[j] for j in range(n_services)]
            
            offspring.extend([child1, child2])
        
        offspring = offspring[:pop_size]
        
        # Mutation (μ = 0.02)
        for chrom in offspring:
            for j in range(n_services):
                if np.random.rand() < 0.02:
                    chrom[j] = np.random.choice(instance_names)
        
        population = offspring
    
    return best_solution, best_fitness, convergence

def recommend_baseline(services: List[Dict]) -> Tuple[List[str], float]:
    """Baseline recommendation: Use m5.large for everything (traditional approach)"""
    baseline = ['m5.large'] * len(services)
    baseline_cost = fitness_function(baseline, services)
    return baseline, baseline_cost

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Docker Compose → AWS Optimizer", page_icon="🐳", layout="wide")

st.title("🐳 Docker Compose to AWS Instance Optimizer")
st.markdown("### Powered by Genetic Algorithm Research (1,050 experiments validated)")

st.info("""
**📊 Based on published research:**
- Optimal GA parameters: pC=1.0, μ=0.02, tournament selection
- Proven to save 23-25% on cloud costs
- Validated with 1,050 experiments (p<0.001, Cohen's d=3.41)
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_iterations = st.slider("GA Iterations", 10, 100, 50, help="More iterations = better results but slower")
    pop_size = st.slider("Population Size", 20, 100, 50, help="From research: 50 is optimal")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This tool uses the **Binary Genetic Algorithm** configuration 
    validated through rigorous research:
    
    - **1,050 experiments**
    - **30 runs per config**
    - **Statistical validation** (ANOVA, Bonferroni)
    - **Large effect sizes** (Cohen's d ≥ 2.62)
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input: Your docker-compose.yml")
    
    example_compose = """services:
  web:
    image: nginx:latest
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
  
  api:
    image: node:18
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1024M
  
  database:
    image: postgres:14
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4096M
  
  cache:
    image: redis:7
    deploy:
      resources:
        limits:
          cpus: '0.25'
          memory: 256M
  
  worker:
    image: python:3.11
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 2048M
"""
    
    if st.button("📋 Load Example"):
        st.session_state.compose_text = example_compose
    
    compose_input = st.text_area(
        "Paste your docker-compose.yml here:",
        value=st.session_state.get('compose_text', ''),
        height=400,
        help="Include the services section with resource limits"
    )
    
    if st.button("🚀 Optimize My Stack", type="primary"):
        if not compose_input.strip():
            st.error("Please paste a docker-compose.yml file")
        else:
            # Parse services
            services = parse_docker_compose(compose_input)
            
            if services:
                st.session_state.services = services
                st.session_state.optimized = False

with col2:
    st.subheader("✨ Results: Optimized Instance Allocation")
    
    if 'services' in st.session_state and st.session_state.services:
        services = st.session_state.services
        
        # Show parsed services
        st.success(f"✅ Detected {len(services)} services")
        
        with st.expander("📊 Parsed Service Requirements"):
            df_services = pd.DataFrame(services)
            st.dataframe(df_services, use_container_width=True)
        
        # Run optimization
        if not st.session_state.get('optimized', False):
            with st.spinner("🔬 Running Genetic Algorithm with optimal parameters..."):
                # Get baseline
                baseline_instances, baseline_cost = recommend_baseline(services)
                
                # Run GA optimization
                optimal_instances, optimal_cost, convergence = binary_ga_optimize(
                    services, 
                    max_iterations=max_iterations,
                    pop_size=pop_size
                )
                
                st.session_state.baseline_instances = baseline_instances
                st.session_state.baseline_cost = baseline_cost
                st.session_state.optimal_instances = optimal_instances
                st.session_state.optimal_cost = optimal_cost
                st.session_state.convergence = convergence
                st.session_state.optimized = True
        
        if st.session_state.get('optimized', False):
            # Display results
            baseline_cost = st.session_state.baseline_cost
            optimal_cost = st.session_state.optimal_cost
            savings_pct = ((baseline_cost - optimal_cost) / baseline_cost) * 100
            
            # Metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("💰 Optimized Cost", f"${optimal_cost:.4f}/hr", 
                         delta=f"-{savings_pct:.1f}%", delta_color="inverse")
            
            with metric_col2:
                st.metric("📊 Baseline Cost", f"${baseline_cost:.4f}/hr", 
                         help="Traditional approach: m5.large for all services")
            
            with metric_col3:
                monthly_savings = (baseline_cost - optimal_cost) * 24 * 30
                st.metric("💵 Monthly Savings", f"${monthly_savings:.2f}")
            
            # Recommendations table
            st.markdown("### 🎯 Instance Recommendations")
            
            recommendations = []
            for service, opt_inst, base_inst in zip(
                services, 
                st.session_state.optimal_instances,
                st.session_state.baseline_instances
            ):
                opt_info = AWS_INSTANCES[opt_inst]
                base_info = AWS_INSTANCES[base_inst]
                
                recommendations.append({
                    'Service': service['name'],
                    'CPU Need': f"{service['cpu']:.2f}",
                    'RAM Need (GB)': f"{service['memory']:.2f}",
                    '✅ Optimal': opt_inst,
                    'Cost/hr': f"${opt_info['cost']:.4f}",
                    '❌ Baseline': base_inst,
                    'Old Cost/hr': f"${base_info['cost']:.4f}",
                })
            
            df_recommendations = pd.DataFrame(recommendations)
            st.dataframe(df_recommendations, use_container_width=True)
            
            # Convergence chart
            st.markdown("### 📈 GA Convergence")
            st.line_chart(st.session_state.convergence)
            st.caption("Cost optimization over iterations - faster convergence = better GA parameters")
            
            # Download button
            st.markdown("### 📥 Export Configuration")
            
            export_text = f"""# AWS Instance Recommendations
# Generated by Docker Compose Optimizer
# Savings: {savings_pct:.1f}% vs baseline

"""
            for service, instance in zip(services, st.session_state.optimal_instances):
                inst_info = AWS_INSTANCES[instance]
                export_text += f"""
# Service: {service['name']}
Instance: {instance}
vCPU: {inst_info['vcpu']}
RAM: {inst_info['ram']} GB
Cost: ${inst_info['cost']:.4f}/hr
Required CPU: {service['cpu']} cores
Required RAM: {service['memory']:.2f} GB

"""
            
            export_text += f"""
# Total Hourly Cost: ${optimal_cost:.4f}
# Monthly Cost (730 hrs): ${optimal_cost * 730:.2f}
# Monthly Savings vs Baseline: ${monthly_savings:.2f}
"""
            
            st.download_button(
                label="📄 Download Recommendations",
                data=export_text,
                file_name="aws_instance_recommendations.txt",
                mime="text/plain"
            )
    else:
        st.info("👈 Paste your docker-compose.yml and click 'Optimize My Stack'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Research-Backed Optimization</strong></p>
    <p>Based on: "Binary Genetic Algorithm for Cost-Optimal Workload Scheduling in Cloud Environments"</p>
    <p>Validated with 1,050 experiments • Statistical significance p<0.001 • Cohen's d=3.41</p>
</div>
""", unsafe_allow_html=True)
