require 'torch'

local function precisionrecallvector(conf, labels, nfalseneg, recallstep)
   assert(conf:nDimension()==1)
   assert(labels:nDimension()==1)
   assert(conf:isSameSizeAs(labels))
   
   local nfalseneg = nfalseneg or 0
   local recallstep = recallstep or 0.1
   
   local so,sortind = scores:sort(true)
   
   local _tp=labels:index(1,sortind):gt(0):float()
   local _fp=labels:index(1,sortind):lt(0):float()
   local npos=labels:eq(1):float():sum() + nfalseneg
   
   -- precision / recall computation
   
   local tp = _tp:cumsum()
   local fp = _fp:cumsum()
   
   local rec = tp/npos
   local _fptp=fp+tp
   local prec = tp:cdiv(_fptp)
   
   -- ap calculation
   
   local ap = 0
   local recallpoints = 0
   local mask
   local p
      
   for i=0,1,recallstep do
      recallpoints=recallpoints+1
   end
   
   for i=0,1,recallstep do
      mask = rec:ge(i)
      if mask:max()>0 then
         p = prec:maskedSelect(mask):max()
      else
         p=0
      end
      ap = ap + p/recallpoints
   end   
   
   return rec, prec, ap, sortind
end


local function precisionrecallmatrix(conf, labels, nfalseneg, recallstep)
   assert(conf:nDimension()==2)
   assert(labels:nDimension()==2)
   assert(conf:isSameSizeAs(labels))
   assert(nfalseneg==nil or nfalseneg:nDimension()==1)
   
   local nSamples = conf:size(2)
   local nClasses = conf:size(1)
   
   -- allocate
   local rec = torch.FloatTensor(nClasses, nSamples)
   local prec = torch.FloatTensor(nClasses, nSamples)
   local ap = torch.FloatTensor(nClasses)
   local sortind = torch.LongTensor(nClasses, nSamples)
   
   for i=1,nClasses do
      local _conf = conf:select(1,i)
      local _labels = labels:select(1,i)
      local _nfalseneg
      if nfalseneg then
         _nfalseneg = nfalseneg[i]
      end
      local _recallstep = recallstep
      local _rec, _prec, _ap, _sortind = precisionrecallvector(_conf, _labels, _nfalseneg, _recallstep)
      
      print(_rec:size())
      print(rec:size())
      rec:select(1,i):copy(_rec)
      prec:select(1,i):copy(_prec)
      ap[i]=_ap
      sortind:select(1,i):copy(_sortind)
   end
   return rec, prec, ap, sortind
end


local function precisionrecall(conf, labels, nfalseneg, recallstep)
   if conf:nDimension()==2 then
      return precisionrecallmatrix(conf, labels, nfalseneg, recallstep)
   elseif conf:nDimension()==1 then
      return precisionrecallvector(conf, labels, nfalseneg, recallstep)
   else
      error('vectors or matrices (classes x samples) expected')
   end
end


return precisionrecall
