import torch

import numpy as np

import faiss

# ------------------------------------------
# faiss implementation

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d
    
    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)
        
    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    faiss.bruteForceKnn(res, metric,
    #faiss.bfKnn(res, metric,
                        xb_ptr, xb_row_major, nb,
                        xq_ptr, xq_row_major, nq,
                        d, k, D_ptr, I_ptr)

    return D, I

def nearest_neighbor_interp_fe(xy_A_b,xy_B_b,R_B_b):
    '''
    Input: xy_A_b bxNx2 matrix where b is batch size, N is regular grid mesh size
           xy_B_b  bxMx2 matrix where b is batch size, M is point cloud size
           R_B_b   bxkxM matrix where b is batch size, M is point cloud size
    Output: R_A_b  bxkxN interpolated value at grid point
    '''
    b,k,M = R_B_b.shape
    _,N,_ = xy_A_b.shape
    # minimum distance by faiss

    #method = "FlatIP"
    method = "IVF"
    
    id_min = torch.zeros([b,N],dtype=torch.int64).cuda()
    
    for n in range(b):

        points_A = xy_A_b[n,:,:].detach().contiguous()
        points_B = xy_B_b[n,:,:].detach().cpu().numpy()
        
        if method == "FlatIP":
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, 2)
            index.add(points_B)
        elif method == "IVF":
            res = faiss.StandardGpuResources()
            nlist = 100
            quantizer = faiss.IndexFlatL2(2)  # the other index
            index_ivf = faiss.IndexIVFFlat(quantizer, 2, nlist, faiss.METRIC_L2)
            # here we specify METRIC_L2, by default it performs inner-product search
        
            # make it an IVF GPU index
            index = faiss.index_cpu_to_gpu(res, 0, index_ivf)
    
            assert not index.is_trained
            index.train(points_B)        # add vectors to the index
            assert index.is_trained
            index.add(points_B)          # add vectors to the index        

        # (1) search_index_pytorch
        # query is pytorch tensor (GPU)
        # no need for a sync here
        D, ID = search_index_pytorch(index, points_A, 1)
        res.syncDefaultStreamCurrentDevice()
        #D,ID = search_raw_array_pytorch(res,xy_B_b[n,:,:],xy_A_b[n,:,:],1)
        id_min[n,:] = ID[:,0]
        
    # add dimension 
    id_min = torch.stack(k*[id_min],axis=1)
    # interpolate from point cloud to grid
    R_A_b = torch.gather(R_B_b,2,id_min)
    return R_A_b

def set_index_faiss(points_B):
    # set FAISS index
    # points_B : numpy array with Mx2 dimension
    
    points_B = np.ascontiguousarray(points_B)
    
    #method = "FlatIP"
    method = "IVF"
    if method == "FlatIP":
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, 2)
        index.add(points_B)
    elif method == "IVF":
        res = faiss.StandardGpuResources()
        nlist = 100
        quantizer = faiss.IndexFlatL2(2)  # the other index
        index_ivf = faiss.IndexIVFFlat(quantizer, 2, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search
        # make it an IVF GPU index
        index = faiss.index_cpu_to_gpu(res, 0, index_ivf)

        assert not index.is_trained
        index.train(points_B)        # add vectors to the index
        assert index.is_trained
        index.add(points_B)          # add vectors to the index
    return index

def nearest_neighbor_interp_fi(xy_A_b,xy_B_b,R_B_b,index,mode):
    '''
    Interpolation by faiss with "precalculated" index
    Input: xy_A_b bxNx2 matrix where b is batch size, N is regular grid mesh size
           xy_B_b  bxMx2 matrix where b is batch size, M is point cloud size
           R_B_b   bxkxM matrix where b is batch size, M is point cloud size
           index  faiss index
           mode   forward or backward
    Output: R_A_b  bxkxN interpolated value at grid point
    '''
    b,k,M = R_B_b.shape
    _,N,_ = xy_A_b.shape
    # minimum distance by faiss
    #print("R_B_b shape",R_B_b.shape)

    id_min = torch.zeros([b,N],dtype=torch.int64).cuda()
    
    for n in range(b):

        if mode=="forward":
            # forward : use normal query 
            points_A = xy_A_b[n,:,:].detach().contiguous()
            D, ID = search_index_pytorch(index, points_A, 1)
            #res.syncDefaultStreamCurrentDevice()
            #print("ID shape fwd",ID.shape)
            
        elif mode=="backward":
            # backward : use reverse query with forward index
            points_B = xy_B_b[n,:,:].detach().contiguous()
            k_rev = 5
            DR, IDR = search_index_pytorch(index, points_B, k_rev)
            # create distance table
            D_tbl = torch.full((N,k_rev),999.9).cuda()
            ID_tbl = torch.zeros(N,k_rev,dtype=torch.int64).cuda()
            # fill distance table
            for kk in range(k_rev):
                D_tbl[IDR[:,kk],kk] = DR[:,kk]
                ID_tbl[IDR[:,kk],kk] = torch.arange(M).cuda()
            min_index = torch.argmin(D_tbl,dim=1,keepdim=True)
            ID = ID_tbl.gather(1,min_index)
            #import pdb;pdb.set_trace()
            #print("ID shape back",ID.shape)
            
        id_min[n,:] = ID[:,0]
        
    # add dimension 
    id_min = torch.stack(k*[id_min],axis=1)
    # interpolate from point cloud to grid
    #print("id_min shape",id_min.shape)
    R_A_b = torch.gather(R_B_b,2,id_min)
    return R_A_b
