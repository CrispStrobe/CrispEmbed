"""Exact integer RGB diagnostics. No tolerance or model-quality gate."""
import math


def small(left,right,width):
    if not left or len(left)!=len(right)or len(left)%3 or width<1 or len(left)%(3*width):raise ValueError('exact RGB geometry required')
    differences=[int(y)-int(x)for x,y in zip(left,right)];channels={}
    summarize=lambda d:dict(max_absolute=max(map(abs,d)),mean_absolute=sum(map(abs,d))/len(d),rms=math.sqrt(sum(x*x for x in d)/len(d)))
    for i,name in enumerate('RGB'):channels[name]=summarize(differences[i::3])
    at=next((i for i,d in enumerate(differences)if d),None)
    first=None if at is None else dict(byte_offset=at,pixel_index=at//3,x=(at//3)%width,y=(at//3)//width,channel='RGB'[at%3],
                 pillow_pixel=list(left[at//3*3:at//3*3+3]),stb_pixel=list(right[at//3*3:at//3*3+3]),signed_difference=differences[at])
    return dict(**summarize(differences),per_channel=channels,first_difference=first,
                changed_values=sum(x!=0 for x in differences),changed_pixels=sum(any(differences[i:i+3])for i in range(0,len(differences),3)),
                rgb_exact=not any(differences),signed_difference='STB minus Pillow')


def files(left,right,width,height):
    import numpy as np
    a=np.memmap(left,dtype='u1',mode='r');b=np.memmap(right,dtype='u1',mode='r')
    if len(a)!=len(b)or len(a)!=width*height*3:raise ValueError('exact raw RGB extents required')
    delta=b.astype(np.int16)-a.astype(np.int16);absolute=np.abs(delta)
    def summarize(d):
        d=d.astype(np.float64)
        return dict(max_absolute=int(np.abs(d).max()),mean_absolute=float(np.abs(d).mean()),rms=float(np.sqrt((d*d).mean())))
    changed=int(np.count_nonzero(delta));first=None
    if changed:
        at=int(np.argmax(delta!=0));p=at//3*3
        first=dict(byte_offset=at,pixel_index=at//3,x=(at//3)%width,y=(at//3)//width,channel='RGB'[at%3],
                   pillow_pixel=a[p:p+3].tolist(),stb_pixel=b[p:p+3].tolist(),signed_difference=int(delta[at]))
    return dict(**summarize(delta),per_channel={n:summarize(delta[i::3])for i,n in enumerate('RGB')},first_difference=first,
                changed_values=int(np.count_nonzero(delta)),changed_pixels=int(np.count_nonzero(np.any(delta.reshape(-1,3),axis=1))),
                rgb_exact=not changed,signed_difference='STB minus Pillow')
