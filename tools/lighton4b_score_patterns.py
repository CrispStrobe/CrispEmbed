"""Model-free score attribution diagnostics; no quality or stage claims."""
import heapq
import math
import struct

def patterns(original,native,top_k=8):
    if len(original)!=len(native)or len(original)%4 or not original:raise ValueError('paired F32 byte geometry differs')
    count=len(original)//4;x=struct.unpack('<%df'%count,original);y=struct.unpack('<%df'%count,native)
    if not all(math.isfinite(v)for v in x+y):raise ValueError('nonfinite score')
    mx=math.fsum(x)/count;my=math.fsum(y)/count
    variance=math.fsum((v-mx)**2 for v in x);covariance=math.fsum((a-mx)*(b-my)for a,b in zip(x,y))
    slope=covariance/variance if variance else 0.0;intercept=my-slope*mx
    residual=[b-(slope*a+intercept)for a,b in zip(x,y)];delta=[b-a for a,b in zip(x,y)];mean_delta=my-mx
    ox=max(range(count),key=x.__getitem__);ny=max(range(count),key=y.__getitem__)
    def summary(values):
        indices=heapq.nlargest(min(top_k,count),range(count),key=values.__getitem__);maximum=values[indices[0]];denominator=math.fsum(math.exp(v-maximum)for v in values)
        return [{'id':i,'score':values[i],'probability':math.exp(values[i]-maximum)/denominator}for i in indices]
    return {'count':count,'original_top':summary(x),'native_top':summary(y),
        'original_winner_native_rank':1+sum(v>y[ox]for v in y),'native_winner_original_rank':1+sum(v>x[ny]for v in x),
        'native_minus_original_winner_gap':{'original':x[ny]-x[ox],'native':y[ny]-y[ox]},
        'raw_max_abs':max(map(abs,delta)),'raw_rms':math.sqrt(math.fsum(v*v for v in delta)/count),
        'mean_shift':mean_delta,'centered_delta_rms':math.sqrt(math.fsum((v-mean_delta)**2 for v in delta)/count),
        'affine_fit':{'native_approximately_slope_times_original_plus_intercept':True,'slope':slope,'intercept':intercept,
            'residual_rms':math.sqrt(math.fsum(v*v for v in residual)/count),'residual_max_abs':max(map(abs,residual)),
            'residual_at_original_winner':residual[ox],'residual_at_native_winner':residual[ny]},
        'limits':'Output-score diagnostic only; affine fitting changes no arrays, logits, thresholds or inference. No internal-stage attribution.'}
