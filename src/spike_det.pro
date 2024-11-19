pro spike_det




file='NB3_img.fits'
mreadfits,file,ind,img

data0=img


sz=size(data0)
data1=fltarr(sz(1)+20,sz(2)+20)
data1(10:10+sz(1)-1,10:10+sz(2)-1)=data0
k=where(data1 lt 0)
data1[k]=0.
x3=[-1,0,1,-1,1,-1,0,1]
y3=[1,1,1,0,0,-1,-1,-1]
spk_img=fltarr(sz(1),sz(2))
spk_img0=fltarr(sz(1)+20,sz(2)+20)
peri=fltarr(8)

th1=400 ;min intensity threshold threshold
th2=1.2; median threshold as in AIA image


progress,0.0,/reset,label='spk detection (%)'
for i=10,10+sz(1)-1 do begin
for j=10,10+sz(2)-1 do begin
int=data1[i,j]
if int gt 0 then begin
im=data1[x3+i,y3+j]
c=where(im gt 0,num)
av=total(im[c])/float(num)
;mn=median(im[c])
if int gt av+th1 and int gt av*th2   then begin
spk_img[i-10,j-10]=1.
spk_img0[i,j]=1.
mn=median(im[c])
data0[i-10,j-10]=mn
;print,i,j
endif
endif
endfor
pct=float(i)*100./float(sz(1))
progress,pct
endfor
progress,100.,/last
c=where(spk_img eq 1.,num)

print,'Number of detected spike =',num

save,file='spike_location.sav',spk_img


;-------spike replacement
;spike replaced with mean of 8 perimeter intensities
;if perimeter includes another spike will not be consired while
;averageing

sz=size(data0)
data2=fltarr(sz(1)+20,sz(2)+20)
data2(10:10+sz(1)-1,10:10+sz(2)-1)=data0

c2=where(spk_img0 eq 1)
data2[c2]=-200
sz=size(data2)

for i=0,sz(1)-1 do begin
for j=0,sz(2)-1 do begin
if spk_img0[i,j] eq 1 then begin
im=data2[x3+i,y3+j]
int=average(im,missing=-200)
data2[i,j]=int
endif
endfor
endfor

data3=data2[10:10+4095,10:10+4095]
spk_rm_img=data3

save,file='spike_rm_data.sav',spk_rm_img


;--





stop
end
