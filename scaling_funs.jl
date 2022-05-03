using Statistics

function rsc(data;n=5)
    r,c=size(data)
    rsc_data=zeros(Float32,(r,c))
    means=zeros(Float32,(r,c))
    stds=zeros(Float32,(r,c))
    for j = 1:c
      for i = 1:r-n
        rsc_data[i,j] = (data[i+n,j]-mean(data[i:i+n,j]))/std(data[i:i+n,j])
        # stds[i,j] = std(rsc_data[i:i+n,j])
      end
    end
    for j = 1:c
      for i = 1:r-2*n
        means[i,j] = mean(rsc_data[i:i+n,j])
        stds[i,j] = std(rsc_data[i:i+n,j])
      end
    end
    return rsc_data[1:end-2*n,:],means[1:end-2*n,:],stds[1:end-2*n,:],data[1+n:r-n,1]
  end

  function sc(x)
    y = (values(x).-mean(values(x)))/std(values(x))
  end