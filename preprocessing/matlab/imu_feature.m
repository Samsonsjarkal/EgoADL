function [time_stamp, imu_feature_buffer] = imu_feature(filename)
%     disp(filename);
    
    T = readtable(filename);
    time_stamp = T{:,1};
    x = T{:,2};
    y = T{:,3};
    z = T{:,4};
    imu_feature_buffer = zeros(size(time_stamp,1), 3);
    imu_feature_buffer(:,1) = x;
    imu_feature_buffer(:,2) = y;
    imu_feature_buffer(:,3) = z;
    
    return
