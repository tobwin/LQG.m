classdef LQG < matlab.mixin.Copyable
    % LQG: copyable handle class for time-varying Linear-Quadratic-Gaussian estimation and control
    % (discrete-time, finite horizon):
    %
    %     Dynamics: x{t+1} = A{t}*x{t} + B{t}*L{t}*x1{t} + v{t}
    %     Feedback: y{t} = C{t}*x{t} + w{t}
    %     Cost: J(t) = x{t}'*Q{t}*x{t} + u{t}'*R{t}*u{t}
    %
    %     x{1} ~ N(x0{1},X*X')
    %     v{t} ~ N(0,V*V')
    %     w{t} ~ N(0,W*W')
    %
    %     Here, x0{t} and x{t} are the a-prioi and a-posteriori state estimates obtained through Kalman-filtering.
    %
    % ---------- System Definition ----------
    %
    % lqg = LQG(T, 'A', A) initializes an LQG object with time horizon T and state transition matrices A{t}.
    %
    % lqg.define('A',A) defines the cell array lqg.A to represent
    %     time-varying state transition matrices A{t}. A can be a cell array
    %     of T-1 matrices A{t} s.t. size(A{t+1},1) = size(A{t},2), or a
    %     matrix. If A is a matrix, then lqg.A{t} = lqg.A{t+1} for all t.
    %
    %     B{t}, C{t}, R{t}, V{t}, W{t} are defined analogously. Defaults
    %     are 0.
    %
    %     Several identical matrices can be defined using, e.g., lqg.define('ABCQRVWX',M).
    %
    %     Note: - X must be defined to be a matrix.
    %           - If Q is a matrix, lqg.define('Q',Q) will define lqg.Q{T} = Q, and
    %             lqg.Q{t} = zeros(size(A{t},2)) for all t<T.
    %
    %
    % ---------- Sampling ----------
    %
    % lqg.sample(N) samples N system trajectories.
    %
    % lqg.sample(N,'x',x), where x is a n times N matrix with initial state dimension n, draws N samples with
    %     initial states x{1} = repmat(x0,1,N) + X*x.
    %
    % lqg.sample(N,'v',v,'w',w) draws N samples with specified v{t} = V*v(:,:,t) and w{t} = W*w(:,:,t).
    %
    % lqg.sample(N,'u',u) draws samples with open loop controls u{t}.
    %
    % lqg.sample(N,'y',y) computes state estimates and predictions x1{t} and x0{t+1}, as well as (hypothetical) controls u{t}.
    %
    %
    % ---------- Statistics ----------
    %
    % lqg.mean returns the expected trajectories of x, y and u
    %
    % lqg.cov returns the covariance matrices of x, y and u
    %
    % lqg.value returns the expected quadratic cost
    %
    %
    % Tobias Winner, 15.09.2018
    % email: winner.tobias@gmail.com
    
    properties ( SetAccess=private )
        horizon
    end
    
    properties
        % system parameters
        A, B, C
        Q, R
        V, W, X
        x
    end
    
    properties ( Hidden, SetAccess=private )
        % custom control policy
        M
    end
    
    properties ( Hidden, Dependent )
        % LQR and Kalman filter
        L, P
        K, X0, X1
        % alias for horizon
        T
    end
    
    properties ( Access=private )
        % memory
        L_, P_
        K_, X0_, X1_
    end
    
    methods
        %% system definition
        
        function self = LQG(horizon, varargin)
            % create LQG object
            assert(horizon > 0 & floor(horizon) == horizon, 'Time horizon must be a positive integer.')
            self.horizon = horizon;
            self.define(varargin{:});
        end
        
        function define(self, varargin)
            varstr = varargin{1};
            val = varargin{2};
            for var = varstr
                if strcmp(var,'L')
                    self.M = val;
                else
                    self.(var) = val;
                end
            end
            if nargin > 3
                varargin = {varargin{3:end}};
                self.define(varargin{:})
            end
        end
        
        function T = get.T(self)
            T = self.horizon;
        end
        
        function set.A(self, A)
            assert(~isempty(A), 'Transition matrix A must be defined.')
            self.clear('lqg')
            self.clear('lqe')
            A = self.repcell(A, self.T-1);
            for t = 2:self.T-1
                assert(size(A{t},2)==size(A{t-1},1), 'Dimensions of A are inconsistent.')
            end
            self.A = A;
            try self.B = self.B; catch, self.B = []; end
            try self.C = self.C; catch, self.C = []; end
            try self.Q = self.Q; catch, self.Q = []; end
            try self.V = self.V; catch, self.V = []; end
            try self.X = self.X; catch, self.X = []; end
            try self.x = self.x; catch, self.x = []; end
            try self.M = self.M; catch, self.M = []; end
        end
        
        function set.B(self, B)
            self.clear('lqr')
            if isempty(B)
                self.M = [];
                self.R = [];
            else
                B = self.repcell(B, self.T-1);
                d = self.dims('x');
                for t = 1:self.T-1
                    assert(size(B{t},1)==d(t+1), 'Dimensions of B don''t match x')
                end
            end
            self.B = B;
            try self.R = self.R; catch, self.R = []; end
            try self.M = self.M; catch, self.M = []; end
        end
        
        function set.C(self, C)
            self.clear('lqe')
            if isempty(C)
            else
                C = self.repcell(C, self.T);
                d = self.dims('x');
                for t = 1:self.T
                    assert(size(C{t},2)==d(t), 'Dimensions of C don''t match x')
                end
            end
            self.C = C;
            try self.W = self.W; catch, self.W = []; end
        end
        
        function set.x(self, x)
            if ~isempty(x)
                assert(iscolumn(x), 'x must be a clomumn vector.')
                assert(length(x)==self.dims('x',1), 'Inconsistent initial state dimension')
            end
            self.x = x;
        end
        
        function set.X(self,X)
            self.clear('lqe')
            if ~isempty(X)
                assert(ismatrix(X), 'X must be a matrix.')
                assert(size(X,1)==self.dims('x',1), 'Inconsistent initial state dimension')
            end
            self.X = X;
        end
        
        function set.Q(self, Q)
            self.clear('lqr')
            if ~isempty(Q)
                Q = self.repcell(Q, self.T, -1);
                d = self.dims('x');
                for t = 1:self.T
                    assert(size(Q{t},2)==d(t), 'Dimensions of Q don''t match x')
                    assert(size(Q{t},2) == size(Q{t},2), 'Q must be symmetric.')
                    assert(all(eig(Q{t})>=-eps), 'Q must be positive semi-definite.')
                end
            end
            self.Q = Q;
        end
        
        function set.R(self, R)
            self.clear('lqr')
            if ~isempty(R)
                R = self.repcell(R, self.T-1);
                d = self.dims('u');
                for t = 1:self.T-1
                    assert(size(R{t},2)==d(t), 'Dimensions of R don''t match u')
                    assert(size(R{t},2) == size(R{t},2), 'R must be symmetric.')
                    assert(all(eig(R{t})>=-eps), 'R must be positive semi-definite.')
                end
            end
            self.R = R;
        end
        
        function set.V(self, V)
            self.clear('lqe')
            if ~isempty(V)
                V = self.repcell(V, self.T-1);
                d = self.dims('x');
                for t = 1:self.T-1
                    assert(size(V{t},1)==d(t), 'Dimensions of V don''t match x')
                end
            end
            self.V = V;
        end
        
        function set.W(self, W)
            self.clear('lqe')
            if ~isempty(W)
                W = self.repcell(W, self.T);
                d = self.dims('y');
                for t = 1:self.T
                    assert(size(W{t},1)==d(t), 'Dimensions of W don''t match y')
                end
            end
            self.W = W;
        end
        
        function set.M(self, L)
            if ~isempty(L)
                L = self.repcell(L, self.T-1);
                dx = self.dims('x');
                du = self.dims('u');
                for t = 1:self.T
                    assert(size(L{t},2)==dx(t), 'Dimensions of L don''t match x')
                    assert(size(L{t},1)==du(t), 'Dimensions of L don''t match u')
                end
            end
            self.M = L;
        end
        
        function set.L(self, L)
            self.M = L;
        end
        
        %% LQR and Kalman filter
        
        % LQR
        function compute_lqr(self)
            if ~isempty(self.B)
                Q = self.Q;
                if isempty(self.Q)
                    d = self.dims('x');
                    for t = 1:self.T
                        Q{t} = zeros(d(t));
                    end
                end
                R = self.R;
                if isempty(self.R)
                    d = self.dims('u');
                    for t = 1:self.T-1
                        R{t} = zeros(d(t));
                    end
                end
                [self.L_, self.P_] = self.lqr(self.A, self.B, Q, R);
            end
        end
        
        function L = get.L(self)
            if isempty(self.M)
                if isempty(self.L_)
                    self.compute_lqr;
                end
                L = self.L_;
            else
                L = self.M;
            end
        end
        
        function P = get.P(self)
            if isempty(self.P_)
                self.compute_lqr;
            end
            P = self.P_;
        end
        
        % Kalman filter
        function compute_lqe(self)
            d = self.dims('x');
            A = self.A;
            C = self.C;
            if isempty(C)
                for t = 1:self.T
                    C{t} = zeros(1,d(t));
                end
            end
            W = self.W;
            if isempty(W)
                for t = 1:self.T
                    W{t} = zeros(size(C{t},1),1);
                end
            end
            V = self.V;
            if isempty(V)
                for t = 1:self.T-1
                    V{t} = zeros(d,1);
                end
            end
            X = self.X;
            if isempty(X)
                X = zeros(size(d(1),1));
            end
            [K, self.X0_, self.X1_] = self.lqe(A, C, V, W, X);
            if ~isempty(self.C)
                self.K_ = K;
            end
        end
        
        function K = get.K(self)
            if isempty(self.K_)
                self.compute_lqe;
            end
            K = self.K_;
        end
        
        function X0 = get.X0(self)
            if isempty(self.X0_)
                self.compute_lqe;
            end
            X0 = self.X0_;
        end
        
        function X1 = get.X1(self)
            if isempty(self.X1_)
                self.compute_lqe;
            end
            X1 = self.X1_;
        end
        
        %% sampling
        function data = sample(self, N, varargin)
            % doc
            T = self.T;
            A = self.A;
            
            obs = ~isempty(self.C);
            ctrl = ~isempty(self.B);
            
            if obs
                C = self.C;
                W = self.W;
                K = self.K;
            end
            if ctrl
                B = self.B;
                Q = self.Q;
                R = self.R;
                L = self.L;
            end
            V = self.V;
            X = self.X;
            x = self.x;
            if isempty(x)
                x = zeros(self.a.in(1),1);
            end
            
            % input parser
            parser = inputParser;
            addParameter(parser,'x',{});
            addParameter(parser,'y',{});
            addParameter(parser,'u',{});
            addParameter(parser,'v',{});
            addParameter(parser,'w',{});
            parse(parser,varargin{:})
            
            smply = isempty(parser.Results.y);
            
            y = parser.Results.y;
            u = parser.Results.u;
            
            if smply
                
                if ~isempty(X)
                    if isempty(parser.Results.x)
                        noise.x = randn(size(X,2),N);
                    elseif isequal(parser.Results.x,0)
                        noise.x = zeros(size(X,2),N);
                    else
                        noise.x = parser.Results.x;
                    end
                end
                
                if ~isempty(V)
                    if isempty(parser.Results.v)
                        for t = 1:T-1
                            noise.v{t} = randn(size(V{t},2),N);
                        end
                    elseif isequal(parser.Results.v,0)
                        for t = 1:T-1
                            noise.v{t} = zeros(size(V{t},2),N);
                        end
                    else
                        noise.v = parser.Results.v;
                    end
                end
                
                if obs & ~isempty(self.W)
                    if obs & isempty(parser.Results.w)
                        for t = 1:T
                            noise.w{t} = randn(size(W{t},2),N);
                        end
                    elseif obs & isequal(parser.Results.w,0)
                        for t = 1:T
                            noise.w{t} = zeros(size(W{t},2),N);
                        end
                    elseif obs
                        noise.w = parser.Results.w;
                    end
                end
                
            end
            
            % main loop
            x0 = {repmat(x,1,N)};
            if smply
                if ~isempty(X)
                    x = {x0{1} + X*noise.x};
                else
                    x = {x0{1}};
                end
            end
            for t = 1:T-1
                
                if obs % compute x1{t}, y{1}
                    if smply
                        if ~isempty(W)
                            y{t} = C{t}*x{t} + W{t}*noise.w{t};
                        else
                            y{t} = C{t}*x{t};
                        end
                    end
                    x1{t} = x0{t} + K{t}*(y{t} - C{t}*x0{t});
                else
                    x1{t} = x0{t};
                end
                
                if ctrl % compute x0{t+1}, u{t}
                    u{t} = L{t}*x1{t};
                    x0{t+1} = A{t}*x1{t} + B{t}*u{t};
                else
                    x0{t+1} = A{t}*x1{t};
                end
                
                if smply % compute x{t+1}
                    if ctrl
                        if ~isempty(V)
                            x{t+1} = A{t}*x{t} + B{t}*u{t} + V{t}*noise.v{t};
                        else
                            x{t+1} = A{t}*x{t} + B{t}*u{t};
                        end
                    elseif ~isempty(V)
                        x{t+1} = A{t}*x{t} + V{t}*noise.v{t};
                    else
                        x{t+1} = A{t}*x{t};
                    end
                end
            end
            if obs
                if smply
                    if ~isempty(W)
                        y{T} = C{T}*x{T} + W{T}*noise.w{T};
                    else
                        y{T} = C{T}*x{T};
                    end
                end
                x1{T} = x0{T} + K{T}*(y{T} - C{T}*x0{T});
            else
                x1{T} = x0{T};
            end
            
            % output data
            if smply
                data.x = x;
            end
            if ~isempty(self.C)
                data.y = y;
            end
            if ctrl
                data.u = u;
            end
            if isempty(parser.Results.y) & ~isempty(self.X)
                data.noise.x = noise.x;
            end
            if isempty(parser.Results.y) & ~isempty(self.V)
                data.noise.v = noise.v;
            end
            if isempty(parser.Results.y) & ~isempty(self.W)
                data.noise.w = noise.w;
            end
            data.est.x0 = x0;
            data.est.x1 = x1;
            
            if (~isempty(self.R) | ~isempty(self.Q)) & ctrl
                data.cost.total = zeros(1,N);
                if ~isempty(self.R)
                    data.cost.r = zeros(1,N);
                    for t = 1:self.T-1
                        data.cost.r = data.cost.r + sum((self.symrt(self.R{t})*data.u{t}).^2,1);
                    end
                    data.cost.total = data.cost.total + data.cost.r;
                end
                if ~isempty(self.Q)
                    data.cost.q = zeros(1,N);
                    for t = 1:self.T
                        data.cost.q = data.cost.q + sum((self.symrt(self.Q{t})*data.x{t}).^2,1);
                    end
                    data.cost.total = data.cost.total + data.cost.q;
                end
            end
        end
        
        %% system transformations
        
        function dummy = dummy(self)
            % inserts zeros for undefined matrices
            A = self.A;
            d = self.dims('x');
            B = self.B;
            if isempty(B)
                for t = 1:self.T-1
                    B{t} = zeros(d(t+1),1);
                end
            end
            V = self.V;
            if isempty(V)
                for t = 1:self.T-1
                    V{t} = zeros(d(t+1),1);
                end
            end
            C = self.C;
            W = self.W;
            if isempty(C)
                for t = 1:self.T
                    C{t} = zeros(1,d(t));
                    W{t} = 0;
                end
            elseif isempty(W)
                W{t} = zeros(size(C{t},1),1);
            end
            X = self.X;
            if isempty(self.X)
                X = zeros(d(1),1);
            end
            x = self.x;
            if isempty(self.x)
                x = zeros(d(1),1);
            end
            dummy = LQG(self.T, ...
                'A', A, 'B', B, 'C', C, 'V', V, 'W', W, 'X', X, 'x', x);
            if ~isempty(self.M)
                dummy.define('L',self.M);
            end
            dummy.define('Q',self.Q);
            dummy.define('R',self.R);
        end
        
        
        function [sys] = augment(self,var)
            % system agmentation
            
            lqg = self.dummy;
            A = lqg.A;
            B = lqg.B;
            C = lqg.C;
            L = lqg.L;
            K = lqg.K;
            R = lqg.R;
            Q = lqg.Q;
            V = lqg.V;
            W = lqg.W;
            X = lqg.X;
            
            % x1 augmentation
            if strcmp(var,'x1')
                
                for t = 1:lqg.T-1
                    if t < lqg.T
                        a = size(A{t});
                        F{t} = [A{t} B{t}*L{t};
                            K{t+1}*C{t+1}*A{t} (A{t}+B{t}*L{t})-(K{t+1}*C{t+1}*A{t})];
                        G{t} = [lqg.V{t} zeros(a(1),size(W{t+1},2));
                            K{t+1}*C{t+1}*lqg.V{t} K{t+1}*W{t+1}];
                    end
                    if ~isempty(lqg.Q) & ~isempty(lqg.R)
                        P{t} = blkdiag(lqg.Q{t},L{t}'*R{t}*L{t});
                    elseif ~isempty(lqg.Q)
                        P{t} = blkdiag(lqg.Q{t},zeros(size(lqg.Q{t})));
                    elseif ~isempty(lqg.R)
                        if t > 1
                            n = size(A{t-1},2);
                        else
                            n = size(X,1);
                        end
                        P{t} = blkdiag(zeros(n),L{t}'*R{t}*L{t});
                    else
                        P = [];
                    end
                end
                if ~isempty(lqg.Q)
                    P{lqg.T} = blkdiag(lqg.Q{lqg.T},zeros(size(lqg.Q{lqg.T})));
                elseif ~isempty(P)
                    n = size(A{end},1);
                    P{lqg.T} = blkdiag(zeros(2*n));
                end
                sys = LQG(lqg.T, 'A', F);
                sys.V = G;
                if ~isempty(P)
                    sys.Q = P;
                end
                sys.X = [lqg.X zeros(size(K{1}*W{1}));
                    K{1}*C{1}*lqg.X K{1}*W{1}];
                sys.x = [lqg.x;lqg.x];
                
            elseif strcmp(var,'x0')
                disp('missing')
            end
            
        end
        
        
        %% statistics
        
        function [val] = value(self)
            % expected cost
            
            if ~isempty(self.B)
                sys = self.augment('x1');
                if isempty(self.x)
                    x = [zeros(2*self.a.in(1),1)];
                else
                    x = [self.x;self.x];
                end
            else
                sys = self;
                if isempty(self.x)
                    x = [zeros(self.a.in(1),1)];
                else
                    x = [self.x];
                end
            end
            
            % main loop
            X = sys.X*sys.X';
            val = 0;
            for t = 1:self.T-1
                val = val + x'*sys.Q{t}*x + trace(sys.Q{t}*X);
                x = sys.A{t}*x;
                X = sys.A{t}*X*sys.A{t}' + sys.V{t}*sys.V{t}';
            end
            val = val + x'*sys.Q{self.T}*x + trace(sys.Q{self.T}*X);
        end
        
        % mean
        function [avg] = mean(self)
            data = self.sample(1,'x',0,'v',0,'w',0);
            x = self.time2trl(data.x);
            avg.x = x{1};
            if ~ isempty(self.C)
                y = self.time2trl(data.y);
                avg.y = y{1};
            end
            if ~ isempty(self.B)
                u = self.time2trl(data.u);
                avg.u = u{1};
            end
        end
        
        % covariance
        function [covar] = cov(self)
            
            if ~isempty(self.B);
                sys = augment(self,'x1');
            else
                sys = self.dummy;
                sys = augment(sys,'x1');
            end
            
            for t = 1:self.T
                x{t} = sys.X0{t}(1:size(sys.X0{t},1)/2,1:size(sys.X0{t},1)/2);
                if ~isempty(self.C)
                    y{t} = self.C{t}*x{t}*self.C{t}';
                end
                if ~isempty(self.W)
                    y{t} = y{t} + self.W{t}*self.W{t}';
                end
                if t < self.T & ~isempty(self.L)
                    u{t} = self.L{t}*sys.X0{t}(size(sys.X0{t},1)/2+1:end,size(sys.X0{t},1)/2+1:end)*self.L{t}';
                end
            end
            
            covar.x = x;
            if ~isempty(self.B)
                covar.u = u;
            end
            if ~isempty(self.C)
                covar.y = y;
            end
            
        end
        
        
        %% Miscellaneous core functions
        
        function d = dims(self, var, idx)
            % returns the dimensionality of x, y or u
            if strcmp(var, 'x')
                if nargin == 3
                    if idx == 1
                        d = size(self.A{1},2);
                    else
                        d = size(self.A{idx-1},1);
                    end
                else
                    d = zeros(1,self.horizon);
                    d(1) = size(self.A{1},2);
                    for t = 1:self.T-1
                        d(t+1) = size(self.A{t},1);
                    end
                end
            elseif strcmp(var, 'y')
                if nargin == 3
                    d = size(self.C{idx},1);
                else
                    d = zeros(1,self.horizon);
                    for t = 1:self.T
                        d(t) = size(self.C{t},1);
                    end
                end
            elseif strcmp(var, 'u')
                if nargin == 3
                    d = size(self.B{idx},2);
                else
                    d = zeros(1,self.horizon-1);
                    for t = 1:self.T-1
                        d(t) = size(self.B{t},2);
                    end
                end
            end
        end
        
        function clear(self, gain)
            % clear internal memory for Kalman filter and LQR
            if strcmp('lqr', gain)
                self.L_ = [];
                self.P_ = [];
            elseif strcmp('lqe', gain)
                self.K_ = [];
                self.X0_ = [];
                self.X1_ = [];
            end
        end
        
    end
    
    methods ( Static )
        
        function [C] = repcell(A, T, varargin)
            % transform matrix into cell array of matrices
            if ~iscell(A)
                C = cell(1,T);
                if nargin == 3
                    C(1:T) = {zeros(size(A))};
                    idx = varargin{1};
                    idx(idx < 0) = T + idx(idx < 0) + 1;
                    C(idx) = {A};
                else
                    C(1:T) = {A};
                end
            else
                C = A;
                assert(length(C) == T, 'Time horizons don''t match.')
            end
        end
        
        function [L, P] = lqr(A, B, Q, R)
            % Linear Quadratic Regulator
            T = length(Q);
            P{T} = Q{T};
            for t = T-1:-1:1
                L{t} = -pinv(R{t} + B{t}'*P{t+1}*B{t}) * B{t}' * P{t+1} * A{t};
                P{t} = A{t}' * P{t+1} * (A{t} + B{t} * L{t}) + Q{t};
            end
        end
        
        function [K, X0, X1] = lqe(A, C, V, W, X)
            % Linear Quadratic Estimator (Kalman filter)
            T = length(C);
            X0 = {X*X'};
            for t = 1:T
                K{t} = X0{t} * C{t}' * pinv(C{t}*X0{t}*C{t}' + W{t}*W{t}');
                X1{t} = X0{t} - K{t} * C{t} * X0{t};
                if t < T
                    X0{t+1} = A{t}*X1{t}*A{t}' + V{t}*V{t}';
                end
            end
        end
        
        function [R] = symrt(S)
            [V,D] = eig(S);
            R = real(V*diag(sqrt(diag(D)))*pinv(V));
        end
        
        function [trl] = time2trl(time)
            T = length(time);
            N = size(time{1},2);
            for t = 1:T
                for i = 1:N
                    trl{i}(:,t) = time{t}(:,i);
                end
            end
        end
        
    end
    
end
