require 'torch'

local function precisionrecall(conf, labels, nfalseneg, recallstep)
   local nfalseneg = nfalseneg or 0
   local recallstep = recallstep or 0.1
   
   local so,sortind = scores:sort(true)
   
   local _tp=labels:index(1,sortind):gt(0):float()
   local _fp=labels:index(1,sortind):lt(0):float()
   local npos=labels:eq(1):float():sum() + falseneg
   
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

return precisionrecall
