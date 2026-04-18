#python3 -c "
from datasets import load_dataset

ds = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
buckets = {'<4192': 0, '4192-8192': 0, '8192-16384': 0, '16384-32768': 0, '32768+': 0}
total = 0
for i, ex in enumerate(ds):
    b = len(ex['text'].encode('utf-8'))
    if b < 4192: buckets['<4192'] += 1
    elif b < 8192: buckets['4192-8192'] += 1
    elif b < 16384: buckets['8192-16384'] += 1
    elif b < 32768: buckets['16384-32768'] += 1
    else: buckets['32768+'] += 1
    total += 1
    if total >= 1000000: break

print(f'Total sampled: {total}')
for k,v in buckets.items():
    print(f'{k}: {v} ({100*v/total:.1f}%)')
#" 2>/dev/null

#Results:
#cap     trunc%  kept_bytes%     uniq_tok        uniq_bigram
#4096    17.70   68.07   99.97%  101.26%
#8192    5.21    81.53   99.97%  100.79%
#16384   1.70    89.77   100.00% 100.00%