clear all
%% csireader.m
%
% read and plot CSI from UDPs created using the nexmon CSI extractor (nexmon.org/csi)
% modify the configuration section to your needs
% make sure you run >mex unpack_float.c before reading values from bcm4358 or bcm4366c0 for the first time
%
% the example.pcap file contains 4(core 0-1, nss 0-1) packets captured on a bcm4358
%

%% configuration
CHIP = '4339';          % wifi chip (possible values 4339, 4358, 43455c0, 4366c0)
BW = 80;                % bandwidth
FILE = '../matlab/stable/dataset/smartphone/1634428580233.pcap';% capture file
NPKTS_MAX = 720000;       % max number of UDPs to process

%% read file
HOFFSET = 16;           % header offset
NFFT = BW*3.2;          % fft size
p = readpcap();
p.open(FILE);
n = min(length(p.all()),NPKTS_MAX);
p.from_start();
csi_buff = complex(zeros(n,NFFT),0);
k = 1;
time_stamp = zeros(n,1);
len_incl = [];
while (k <= n)
    f = p.next();
    time_stamp(k) = int64(f.header.ts_sec)*1000 + int64(f.header.ts_usec)/1000.0;
    len_incl = [len_incl, f.header.orig_len];
    if isempty(f)
        disp('no more frames');
        break;
    end
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

%% plot
% plot(abs(csi_buff(:,50)))
% xlabel("Packets");
% ylabel("Subcarrier 50 CSI amplitude");
% 
% plotcsi(csi_buff, NFFT, false);
% 
csi_buff_new = [];
time_stamp_new = [];
csi_buff_new = complex(zeros(n,NFFT),0);
time_stamp_new = zeros(n,1);
now = 0;
for i = 1: size(csi_buff,1)
%     plot(abs(csi_buff(i,:)));
%     disp(abs(csi_buff(i,33)));
%     pause;
%     disp(abs(csi_buff(i,33)));
    if (mean(abs(csi_buff(i,7:32))) > 200 && abs(csi_buff(i,33)) < 200 && mean(abs(csi_buff(i,70:96))) > 200 && abs(csi_buff(i,97)) < 200 && ...
        mean(abs(csi_buff(i,135:160))) > 200 && abs(csi_buff(i,161)) < 200 && mean(abs(csi_buff(i,199:224))) > 200 && abs(csi_buff(i,225)) < 50)
        now = now + 1;
        csi_buff_new(now,:) = csi_buff(i,:);
        time_stamp_new(now) = time_stamp(i);
%         disp('1');
    end
    
end


subplot(211);
plot(abs(csi_buff_new(:,50)));
hold on;
plot(abs(csi_buff_new(:,95)));
xlabel("Packets");
ylabel("Subcarrier 50 CSI amplitude");

% plotcsi(csi_buff_new, NFFT, false);
% 
% subplot(312);
% % time_stamp_new = time_stamp_new - time_stamp_new(1);
% % plot(double(time_stamp_new));
% plot(double(time_stamp_new - time_stamp_new(1)) / 1000.0);
% xlabel("Packets");
% ylabel("Time stamp (s)");

% csi_buff_full = [];
% time_stamp_full = [];
% for i = 1: size(csi_buff,1)
%     plot(abs(csi_buff(i,:)));
%     disp(abs(csi_buff(i,33)));
%     pause;
% %     disp(abs(csi_buff(i,33)));
%     if (abs(csi_buff(i,33)) < 50 && abs(csi_buff(i,97)) < 50 && abs(csi_buff(i,161)) < 50 && abs(csi_buff(i,225)) < 50)
%         csi_buff_full = [csi_buff_full; csi_buff(i,:)];
%         time_stamp_full = [time_stamp_full time_stamp(i)];
%     end
% end

% subplot(413);
% plot(abs(csi_buff_new(:,50)));
% hold on;
% plot(abs(csi_buff_new(:,95)));
% xlabel("Packets");
% ylabel("Subcarrier 50 CSI amplitude");


csi_buff_new = [csi_buff_new(:,7:32), csi_buff_new(:,34:59), csi_buff_new(:,71:96), ...
    csi_buff_new(:,98:123), csi_buff_new(:,135:160), csi_buff_new(:,162:187), ...
    csi_buff_new(:,199:224), csi_buff_new(:,226:251)];

% figure();
% plot(abs(csi_buff_new(1:20,:))');
% xlabel("Subcarriers");
% ylabel("CSI amplitude");
% ylim([0 2000]);

% amplitude_change = [];
% figure();
% subplot(211);
% plot(abs(csi_buff_new(:,50)));
% xlabel("Packets");
% ylabel("Subcarrier 50 CSI amplitude");
% legend("W/o AGC cancellation");
% hold on;

%% Median
for i =1:size(csi_buff_new,1)-1
    now = abs(csi_buff_new(i+1,:)) ./ abs(csi_buff_new(i,:));
    
%     plot(now);
%     pause;
    if (median(now)<0.9)
%         disp(csi_buff_new(i+1,:))
        csi_buff_new(i+1,:) = csi_buff_new(i+1,:) / median(now);
%         disp(i);
%         disp(csi_buff_new(i+1,:))
    elseif (median(now)>1.1)
        csi_buff_new(i+1,:) = csi_buff_new(i+1,:) * median(now);
%         disp(i);
    end    
%     amplitude_change = [amplitude_change, now];
end

%% Sum
csi_buff_new_amp = abs(csi_buff_new);
csi_buff_new_amp = sum(csi_buff_new_amp');
csi_buff_new_result = abs(csi_buff_new) ./ csi_buff_new_amp';


% subplot(212);
% plot(abs(csi_buff_new(:,50)));
% xlabel("Packets");
% ylabel("Subcarrier 50 CSI amplitude");
% 
% legend("W/ AGC cancellation");

% figure();
subplot(212);
plot(double(time_stamp_new - time_stamp_new(1)) / 1000.0, abs(csi_buff_new_result(:,50)));
hold on;
plot(double(time_stamp_new - time_stamp_new(1)) / 1000.0, abs(csi_buff_new_result(:,95)));
xlabel("Packets");
ylabel("CSI amplitude");

% subplot(414);
% % plotcsi(csi_buff_new, NFFT, false);
% % 
% plot(double(time_stamp_new - time_stamp_new(1)) / 1000.0);
% xlabel("Packets");
% ylabel("Time stamp (s)");
% 
% legend("Subcarrier 50", "Subcarrier 95");


