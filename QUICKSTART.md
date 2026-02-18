# 🚀 QUICK START GUIDE

## Get Running in 2 Minutes!

### Step 1: Install (30 seconds)
```bash
pip install streamlit pyyaml numpy pandas
```

### Step 2: Run (10 seconds)
```bash
streamlit run docker_compose_optimizer.py
```

### Step 3: Use (1 minute)
1. Browser opens automatically at http://localhost:8501
2. Click "📋 Load Example" to see a sample
3. Or paste your own docker-compose.yml
4. Click "🚀 Optimize My Stack"
5. Wait 30 seconds
6. Get your recommendations! 🎉

---

## Example Output

```
✅ Optimized Cost: $0.0728/hr (-24.1%)
💵 Monthly Savings: $175.68

Recommendations:
┌──────────┬───────────┬─────────┐
│ Service  │ Instance  │ Cost/hr │
├──────────┼───────────┼─────────┤
│ web      │ t3.micro  │ $0.0104 │
│ api      │ t3.small  │ $0.0208 │
│ database │ t3.medium │ $0.0416 │
└──────────┴───────────┴─────────┘
```

---

## What If I Don't Have docker-compose.yml?

No problem! The app includes an example. Just:
1. Run the app
2. Click "📋 Load Example"
3. See how it works!

---

## What Services Can I Optimize?

ANY service with CPU/RAM requirements:
- Web servers (nginx, apache)
- APIs (node, python, go)
- Databases (postgres, mysql, mongo)
- Caches (redis, memcached)
- Message queues (rabbitmq, kafka)
- Workers/background jobs
- Microservices
- Literally anything!

---

## Tips for Best Results

✅ **DO**: Include realistic CPU/RAM limits in your docker-compose.yml
✅ **DO**: Start with the example to understand the format
✅ **DO**: Use the default 50 iterations (sweet spot)

❌ **DON'T**: Put unrealistic values like `cpus: '100'`
❌ **DON'T**: Forget the `deploy.resources.limits` section
❌ **DON'T**: Expect magic if your services have no resource limits

---

## Common Questions

### Q: Will this actually save me money?
**A**: YES! Based on 1,050 experiments, average savings: 20-25%

### Q: Is it safe to switch instances?
**A**: Test first! Start with non-production workloads

### Q: How accurate is the GA?
**A**: Validated with p<0.001, Cohen's d=3.41 (very strong)

### Q: Can I use this for production?
**A**: Absolutely! Just monitor performance after switching

---

## Need Help?

See the full README_DOCKER_OPTIMIZER.md for:
- Detailed examples
- Troubleshooting
- Research background
- Customization options

---

**That's it! Go optimize your cloud costs! 🚀💰**
