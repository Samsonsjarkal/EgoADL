file_label_path = '';
file_label=fopen(file_label_path,'r');
formatSpec='%c';
A=fscanf(file_label,formatSpec);

label = [];
for i=1:size(A,2)
    if (A(i) == 'c')
%         disp(A(i:i+3));
        label = [label; A(i:i+3)];
    end
end

data_path = '';

vis = 1;
for i=1:size(label,1)
    data_path_i = data_path + string(label(i,:)) + '\';
%     disp(data_path_i);
    data_dir  = dir(data_path_i + '*.pcap');
    id = 0;
    
    for j = 1:size(data_dir)
        csi_filename = [data_dir(j).folder '\' data_dir(j).name];
        imu_filename = [csi_filename(1:size(csi_filename, 2)-5) '.csv'];
        sound_filename = [csi_filename(1:size(csi_filename, 2)-5) '.wav'];
        if (exist(imu_filename) && exist(sound_filename))
            disp(csi_filename);
            id = id+1;
            if (id<=2 || id == 13)
                continue;
            end
            
            %% Extract features
            [csi_time_stamp, csi_feature_buffer] = csi_feature(csi_filename);
            [imu_time_stamp, imu_feature_buffer] = imu_feature(imu_filename);
            [y, Fs] = sound_feature(sound_filename);
            
            %% Interpolation + filter: CSI 1000 Hz, audio 16 kHz, IMU 500 Hz
            for ii = 1:size(csi_time_stamp,2) - 1
                if (csi_time_stamp(ii+1) - csi_time_stamp(ii) <= 0)
                    csi_time_stamp(ii+1) = csi_time_stamp(ii) + 0.5;
                end
            end
            
            
            fs_csi = 1000;
            csi_time_stamp_interp = csi_time_stamp(1):1000.0/fs_csi:csi_time_stamp(end);
            csi_feature_buffer_interp = interp1(csi_time_stamp',abs(csi_feature_buffer),csi_time_stamp_interp','linear');
            csi_feature_buffer_interp_hampel = hampel(csi_feature_buffer_interp,7);
            csi_feature_buffer_interp_hampel = lowpass(csi_feature_buffer_interp_hampel,100,fs_csi);
            
            
            for ii = 1:size(imu_time_stamp,1) - 1
                if (imu_time_stamp(ii+1) - imu_time_stamp(ii) <= 0)
                    imu_time_stamp(ii+1) = imu_time_stamp(ii) + 0.5;
                end
            end
            
            fs_IMU = 200;
            imu_time_stamp_interp = imu_time_stamp(1):1000.0/fs_IMU:imu_time_stamp(end);
            imu_feature_buffer_interp = interp1(imu_time_stamp',abs(imu_feature_buffer),imu_time_stamp_interp','linear');
            
            fs_MIC = 16000;
            mic_time_stamp_interp = imu_time_stamp(end) - 5000 + linspace(0, 5000, size(y,1));
            mic_feature_buffer = y;
            
            
            if (vis == 1)
                subplot(311);
                plot(csi_time_stamp_interp, abs(csi_feature_buffer_interp_hampel(:,1:1)));
                xlim([imu_time_stamp(1), imu_time_stamp(end)]);
                subplot(312);
                plot(imu_time_stamp_interp, imu_feature_buffer_interp(:,1));
                hold on;
                plot(imu_time_stamp_interp, imu_feature_buffer_interp(:,2));
                plot(imu_time_stamp_interp, imu_feature_buffer_interp(:,3));
                xlim([imu_time_stamp(1), imu_time_stamp(end)]);
                subplot(313);
                plot(mic_time_stamp_interp, mic_feature_buffer);
                xlim([imu_time_stamp(1), imu_time_stamp(end)]);
            end
            
            save([csi_filename(1:size(csi_filename, 2)-5) '_csi.mat'],'csi_time_stamp_interp','csi_feature_buffer_interp_hampel');
            save([csi_filename(1:size(csi_filename, 2)-5) '_imu.mat'],'imu_time_stamp_interp','imu_feature_buffer_interp');
            save([csi_filename(1:size(csi_filename, 2)-5) '_mic.mat'],'mic_time_stamp_interp','mic_feature_buffer');
            pause;
        end
    end
end