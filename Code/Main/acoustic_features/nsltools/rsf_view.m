function rsf_view(rsf, rv, sv, paras)
% RSF_VIEW view the rate-scale-frequency plot 
%	rsf_view(rsf, rv, sv, paras);
%	rsf	: scale-rate-freq matrix 
%	rv      : rate vector in Hz, e.g., 2.^(1:.5:5).
%       sv      : sacel vector in cyc/oct, e.g., 2.^(-2:.5:3).
%	paras	: parameters (see WAV2AUD)
%
%	RSF_VIEW views the output has to be generated by COR2RST, AUD2RST.
%	This function will plot the negative rate plot in the left panel,
%	and positive one on the right panel.
%	See also: COR2RST, AUD2COR, AUD2RST

% Auther: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
% v1.00: 06-Jan-99


global TICKLABEL VER;

% dimensions
dim = length(size(rsf));

if (dim==3)
	[K2, K, N] = size(rsf);
	K1	= length(rv);
else
	error('Dimension of rsf must be 3 !')
end

if (length(sv) ~= K2 | K~=2*K1), error('Size mismatch !'); end;
sgnstr	= '-+';

% graphics parameter
NTIMES	= 2;	dK = 2^(-NTIMES);
max_rs	= max(rsf(:));
RSLIM	= max_rs/1.2;
disp(sprintf('Max: %3.1e; Mean: %3.1e', max_rs, mean(rsf(:))));

% xtick labels
xtic = []; xtlabel = []; Kx = 5;
for k = 1:K1,
	R1 = log2(rv(k));
	if abs(R1-round(R1)) < .001,
		xtic = [xtic k];
		xtlabel = [xtlabel; '-', sprintf('%5.1f', rv(k))];
	end;
end;

% ytick (log. frequency axis) labels
ytlabel = [];
for fdx = 5:-1:1,
	ystr = num2str(round(1000*2^(fdx-3+paras(4))));
	L1 = length(ystr);
	L2 = size(ytlabel, 2);
	if L1 < L2, ystr = [32*ones(1, L2-L1) ystr]; end;
	ytlabel= [ystr; ytlabel];
end;

% options
figsize([11 8.5]*0.75); 
uheight = 1/(K2+(K2-1)*.15+2*0.25);
uwidth = 1/(2+0.05+0.2*2);

% detect colormap
A1MAP = isa1map;

for n = 1:K2,
	for sgdx = 1:2,
		% select subplot 
		subplot('position',[(0.2+(sgdx-1)*1.05)*uwidth, ...
			(0.25+(n-1)*1.15)*uheight, uwidth, uheight]);

		% constructing rate-scale matrix
				
		rs0 = rsf(n, (1:K1)+(sgdx-1)*K1, :);
		rs0 = squeeze(rs0)';
		
		rs0 = interp2(rs0, 1:dK:K1, [1:N]', 'cubic');


		% select colormap 
		if A1MAP,
			rs0 = real_col(rs0, RSLIM);
			image(1:K1, 1:N, rs0);
		else,
			imagesc(1:K1, 1:N, rs0, [0 RSLIM]);
		end;
		axis xy;

		if sgdx == 1 
			set(gca, 'Xdir', 'rev');
			ylabel([num2str(sv(n)) ' cyc/oct'],...
				'fontsi',14,'fontw','bold');  
		end
		set(gca, 'xtick', xtic, ...
			['x' TICKLABEL], xtlabel(:, sgdx:Kx), ...
			'ytick', 12:24:128,...
			['y' TICKLABEL], ytlabel,...
			'fontsi', 6);
		if sgdx == 2, 
			set(gca, 'yaxislocation','right');
			ylabel('Freq. (Hz)');
		end
		if n == 1, xlabel('Rate (Hz)','fontsi',10); end
	end;
	drawnow;
end;

