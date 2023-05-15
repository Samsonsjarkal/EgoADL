%% csireader.m
%
% read and plot CSI from UDPs created using the nexmon CSI extractor (nexmon.org/csi)
% modify the configuration section to your needs
% make sure you run >mex unpack_float.c before reading values from bcm4358 or bcm4366c0 for the first time
%
% the example.pcap file contains 4(core 0-1, nss 0-1) packets captured on a bcm4358
%
% Usage: [csi_time_stamp, csi_feature_buffer] = csi_feature(csi_filename);

function [time_stamp_new, csi_buff_new] = csi_feature(filename)


    CHIP = '4339';          % wifi chip (possible values 4339, 4358, 43455c0, 4366c0)
    BW = 80;                % bandwidth
    FILE = filename;% capture file
    NPKTS_MAX = 20000;       % max number of UDPs to process

    %% Read file
    HOFFSET = 16;           % header offset
    NFFT = BW*3.2;          % fft size
    p = readpcap();
    p.open(FILE);
    n = min(length(p.all()),NPKTS_MAX);
    p.from_start();
    csi_buff = complex(zeros(n,NFFT),0);
    k = 1;
    time_stamp = [];
    len_incl = [];
    
    while (k <= n)
        f = p.next();
        if isempty(f)
            disp('no more frames');
            break;
        end
        time_stamp = [time_stamp, double(f.header.ts_sec)*1000 + double(f.header.ts_usec)/1000.0];
        len_incl = [len_incl, f.header.orig_len];
        if f.header.orig_len-(HOFFSET-1)*4 ~= NFFT*4
            disp('skipped frame with incorrect size');
            continue;
        end
        payload = f.payload;
        H = payload(HOFFSET:HOFFSET+NFFT-1);
        if (strcmp(CHIP,'4339') || strcmp(CHIP,'43455c0'))
            Hout = typecast(H, 'int16');
        elseif (strcmp(CHIP,'4358'))
            Hout = unpack_float(int32(0), int32(NFFT), H);
        elseif (strcmp(CHIP,'4366c0'))
            Hout = unpack_float(int32(1), int32(NFFT), H);
        else
            disp('invalid CHIP');
            break;
        end
        Hout = reshape(Hout,2,[]).';
        cmplx = double(Hout(1:NFFT,1))+1j*double(Hout(1:NFFT,2));
        csi_buff(k,:) = cmplx.';
        k = k + 1;
    end
    
    %% Filter the packets
    csi_buff_new = zeros(size(csi_buff), 'like',csi_buff);
    time_stamp_new = zeros(size(time_stamp), 'like', time_stamp);
%     csi_buff_new = [];
%     time_stamp_new = [];
    
    %% 40 MHz
%     id = 0;
%     for i = 1: size(csi_buff,1)
%     %     plot(abs(csi_buff(i,:)));
%     %     disp(abs(csi_buff(i,33)));
%     %     pause;
%     %     disp(abs(csi_buff(i,33)));
%         if (abs(csi_buff(i,160)) > 50 && abs(csi_buff(i,161)) < 50 && abs(csi_buff(i,224)) > 50 && abs(csi_buff(i,225)) < 50)
%             id = id +1;
%             csi_buff_new(id,:) = csi_buff(i,:);
%             time_stamp_new(1,id) = time_stamp(1,i);
%     %         disp('1');
%         end
% %         csi_buff_new = [csi_buff_new; csi_buff(i,:)];
% %         time_stamp_new = [time_stamp_new time_stamp(i)];
%     end
%     csi_buff_new = csi_buff_new(1:id,:);
%     time_stamp_new = time_stamp_new(1,1:id);
    
    %% 80 MHz
    id = 0;
    for i = 1: size(csi_buff,1)
    %     plot(abs(csi_buff(i,:)));
    %     disp(abs(csi_buff(i,33)));
    %     pause;
    %     disp(abs(csi_buff(i,33)));
        if (abs(csi_buff(i,32)) > 50 && abs(csi_buff(i,33)) < 50 && abs(csi_buff(i,96)) > 50 && abs(csi_buff(i,97)) < 50 && ...
            abs(csi_buff(i,160)) > 50 && abs(csi_buff(i,161)) < 50 && abs(csi_buff(i,224)) > 50 && abs(csi_buff(i,225)) < 50)
            id = id + 1;
            csi_buff_new(id,:) = csi_buff(i,:);
            time_stamp_new(1,id) = time_stamp(1,i);
    %         disp('1');
        end
%         csi_buff_new = [csi_buff_new; csi_buff(i,:)];
%         time_stamp_new = [time_stamp_new time_stamp(i)];
    end
    csi_buff_new = csi_buff_new(1:id,:);
    time_stamp_new = time_stamp_new(1,1:id);
    
    time_stamp_new = double(time_stamp_new);
    
    
    %% Extract the subcarriers
    csi_buff_new = [csi_buff_new(:,7:32), csi_buff_new(:,34:59), csi_buff_new(:,71:96), ...
    csi_buff_new(:,98:123), csi_buff_new(:,135:160), csi_buff_new(:,162:187), ...
    csi_buff_new(:,199:224), csi_buff_new(:,226:251)];

    %% AGC cancellation
    AGC_threshold = 0.1;
    for i =1:size(csi_buff_new,1)-1
        now = abs(csi_buff_new(i+1,:)) ./ abs(csi_buff_new(i,:));
        if (median(now)<1 - AGC_threshold)
            csi_buff_new(i+1,:) = csi_buff_new(i+1,:) / median(now);
        elseif (median(now)>1 + AGC_threshold)
            csi_buff_new(i+1,:) = csi_buff_new(i+1,:) / median(now);
        end    
    end
    
    return
    
